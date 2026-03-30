# -*- coding: utf-8 -*-
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import optuna
import time
import torch
import warnings
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# Игнорируем лишние предупреждения
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- КОНФИГУРАЦИЯ ---
PROJECT_DIR = 'project_files'  # Папка для логов и баз данных
DATA_FILE = 'million.csv'  # Файл с данными
FINAL_MODEL_NAME = 'final_model_god_v4.joblib'
LOG_FILE = 'training_log.txt'

# Параметры обучения
N_TRIALS_FAST = 100  # Много попыток для быстрых (LGBM, XGBoost)
N_TRIALS_SLOW = 40  # Поменьше для медленного (CatBoost)
OPTUNA_SAMPLE = 50000  # Размер выборки для поиска
N_CLUSTERS = 15  # Кластеры для генерации признаков
QUANTILE_ALPHA = 0.50  # Медиана
RANDOM_STATE = 42

# Создаем рабочую директорию
os.makedirs(PROJECT_DIR, exist_ok=True)


# Настройка Логгера
def setup_logger():
    logger = logging.getLogger('EntropyLogger')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(PROJECT_DIR, LOG_FILE), encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
    return logger


log = setup_logger()

# Проверка GPU
TRY_GPU = torch.cuda.is_available()
log.info(f"🖥️ GPU Status: {'AVAILABLE 🔥' if TRY_GPU else 'NOT FOUND ❄️ (Using CPU)'}")

# --- МАТЕМАТИКА ---
EPSILON = 1e-4


def to_logit(y):
    y_safe = np.clip(y, EPSILON, 1 - EPSILON)
    return np.log(y_safe / (1 - y_safe))


def from_logit(z):
    res = 1 / (1 + np.exp(-z))
    return np.clip(res, 0, 1)


def calculate_weights(y):
    y_safe = np.clip(y, 0, 0.99)
    weights = 1 + np.log(1 / (1 - y_safe))
    weights = np.clip(weights, 1, 10)
    return weights / weights.mean()


def pinball_loss(y_true, y_pred, alpha):
    residual = y_true - y_pred
    return np.mean(np.maximum(alpha * residual, (alpha - 1) * residual))


# --- ПРЕПРОЦЕССИНГ ---
class SmartPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=15):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
        self.n_clusters = n_clusters

    def fit(self, X_raw, y=None):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        self.scaler.fit(X_log)
        X_scaled = self.scaler.transform(X_log)
        self.kmeans.fit(X_scaled)
        return self

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)

        df['row_mean'] = df.mean(axis=1)
        df['row_std'] = df.std(axis=1)

        dists = self.kmeans.transform(X_scaled)
        for i in range(self.n_clusters):
            df[f'dist_clust_{i}'] = dists[:, i]

        df['cluster_id'] = self.kmeans.predict(X_scaled)
        return df


class IsotonicCorrector:
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    def fit(self, y_pred, y_true):
        self.iso.fit(y_pred, y_true)
        return self

    def predict(self, y_pred):
        return self.iso.predict(y_pred)


# --- ФУНКЦИИ OPTUNA ---
def get_params(trial, model_name):
    if model_name == 'xgb':
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 600, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'verbosity': 0,
            'tree_method': 'hist',
            'device': 'cuda' if TRY_GPU else 'cpu'
        }
    elif model_name == 'lgbm':
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 600, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 120),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'device': 'cpu', 'n_jobs': -1, 'verbose': -1
        }
    elif model_name == 'cat':
        p = {
            'iterations': trial.suggest_int('iterations', 600, 1400),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'depth': trial.suggest_int('depth', 4, 9),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
            'task_type': 'CPU', 'thread_count': -1, 'verbose': 0
        }
    return p


def obj_generic(trial, X, y, w, model_name, obj_type):
    X_t, X_v, y_t, y_v, w_t, _ = train_test_split(X, y, w, test_size=0.2, random_state=42)
    p = get_params(trial, model_name)

    try:
        if model_name == 'xgb':
            if obj_type == 'q':
                p.update({'objective': 'reg:quantileerror', 'quantile_alpha': QUANTILE_ALPHA})
                model = xgb.XGBRegressor(**p)
                model.fit(X_t, y_t, sample_weight=w_t)
                loss = pinball_loss(y_v, model.predict(X_v), QUANTILE_ALPHA)
            else:
                p.update({'objective': 'reg:squarederror'})
                model = xgb.XGBRegressor(**p)
                model.fit(X_t, y_t, sample_weight=w_t)
                loss = np.sqrt(mean_squared_error(y_v, model.predict(X_v)))

        elif model_name == 'lgbm':
            if obj_type == 'q':
                p.update({'objective': 'quantile', 'alpha': QUANTILE_ALPHA})
                model = lgb.LGBMRegressor(**p)
                model.fit(X_t, y_t, sample_weight=w_t)
                loss = pinball_loss(y_v, model.predict(X_v), QUANTILE_ALPHA)
            else:
                p.update({'objective': 'regression', 'metric': 'rmse'})
                model = lgb.LGBMRegressor(**p)
                model.fit(X_t, y_t, sample_weight=w_t)
                loss = np.sqrt(mean_squared_error(y_v, model.predict(X_v)))

        elif model_name == 'cat':
            if obj_type == 'q':
                p.update({'loss_function': f'Quantile:alpha={QUANTILE_ALPHA}'})
                model = cb.CatBoostRegressor(**p)
                model.fit(X_t, y_t, sample_weight=w_t)
                loss = pinball_loss(y_v, model.predict(X_v), QUANTILE_ALPHA)
            else:
                p.update({'loss_function': 'RMSE'})
                model = cb.CatBoostRegressor(**p)
                model.fit(X_t, y_t, sample_weight=w_t)
                loss = np.sqrt(mean_squared_error(y_v, model.predict(X_v)))

    except Exception as e:
        if 'xgb' in model_name and 'device' in p and p['device'] == 'cuda':
            return 9999.0
        raise e
    return loss


def run_optuna_local(study_name, func, n_trials):
    db_path = f"sqlite:///{os.path.join(PROJECT_DIR, study_name)}.db"
    study = optuna.create_study(study_name=study_name, storage=db_path, load_if_exists=True, direction="minimize")

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    left = n_trials - completed

    if left > 0:
        log.info(f"🔎 {study_name}: Запуск {left} итераций (Target: {n_trials}, Done: {completed})...")
        with tqdm(total=left) as pbar:
            study.optimize(func, n_trials=left, n_jobs=1, callbacks=[lambda s, t: pbar.update(1)])
    else:
        log.info(f"✅ {study_name}: Уже готово {completed}/{n_trials} итераций.")

    return study.best_params


# --- ГИБРИДНАЯ МОДЕЛЬ ---
class HybridModelGod:
    def __init__(self, p_lgbm_q, p_lgbm_m, p_xgb_q, p_xgb_m, p_cat_q, p_cat_m):
        self.bias_corrector = IsotonicCorrector()
        self.final_preprocessor = None

        SCALE_EST = 1.2
        SCALE_LR = 0.85

        def adj(p, k_est='n_estimators', k_lr='learning_rate'):
            pc = p.copy()
            if k_est in pc: pc[k_est] = int(pc[k_est] * SCALE_EST)
            if k_lr in pc: pc[k_lr] = pc[k_lr] * SCALE_LR
            return pc

        self.configs = [
            ('lgbm_q', lgb.LGBMRegressor,
             {**adj(p_lgbm_q), 'objective': 'quantile', 'alpha': QUANTILE_ALPHA, 'device': 'cpu', 'n_jobs': -1,
              'verbose': -1}),
            ('lgbm_m', lgb.LGBMRegressor,
             {**adj(p_lgbm_m), 'objective': 'regression', 'metric': 'rmse', 'device': 'cpu', 'n_jobs': -1,
              'verbose': -1}),
            ('xgb_q', xgb.XGBRegressor,
             {**adj(p_xgb_q), 'objective': 'reg:quantileerror', 'quantile_alpha': QUANTILE_ALPHA, 'tree_method': 'hist',
              'verbosity': 0}),
            ('xgb_m', xgb.XGBRegressor,
             {**adj(p_xgb_m), 'objective': 'reg:squarederror', 'tree_method': 'hist', 'verbosity': 0}),
            ('cat_q', cb.CatBoostRegressor,
             {**adj(p_cat_q, 'iterations'), 'loss_function': f'Quantile:alpha={QUANTILE_ALPHA}', 'task_type': 'CPU',
              'verbose': 0}),
            ('cat_m', cb.CatBoostRegressor,
             {**adj(p_cat_m, 'iterations'), 'loss_function': 'RMSE', 'task_type': 'CPU', 'verbose': 0}),
        ]

        self.final_estimator = Ridge(alpha=1.0, random_state=RANDOM_STATE)
        self.trained_models = []

    def _fit_single(self, cls, params, X, y, w, name):
        run_params = params.copy()
        try:
            if 'xgb' in name and TRY_GPU:
                run_params['device'] = 'cuda'
                m = cls(**run_params);
                m.fit(X, y, sample_weight=w)
            else:
                if 'xgb' in name: run_params['device'] = 'cpu'
                m = cls(**run_params);
                m.fit(X, y, sample_weight=w)
        except Exception as e:
            log.warning(f"⚠️ GPU Fail in {name}, switching to CPU.")
            if 'xgb' in name:
                run_params['device'] = 'cpu'
                m = cls(**run_params);
                m.fit(X, y, sample_weight=w)
            else:
                raise e
        return m

    def fit(self, X_raw_full, y_full, weights_full):
        kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        meta_features = np.zeros((X_raw_full.shape[0], len(self.configs) + 1))

        log.info("🚀 Training Stacking (CV Phase)...")

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw_full)):
            X_tr_raw = X_raw_full.iloc[train_idx]
            X_val_raw = X_raw_full.iloc[val_idx]
            y_tr = y_full.iloc[train_idx]
            w_tr = weights_full.iloc[train_idx]

            prep = SmartPreprocessor(n_clusters=N_CLUSTERS).fit(X_tr_raw)
            X_tr = prep.transform(X_tr_raw)
            X_val = prep.transform(X_val_raw)

            col = 0
            for name, cls, params in self.configs:
                m = self._fit_single(cls, params, X_tr, y_tr, w_tr, name)
                meta_features[val_idx, col] = m.predict(X_val)
                col += 1

            mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=RANDOM_STATE)
            mlp.fit(X_tr, y_tr)
            meta_features[val_idx, col] = mlp.predict(X_val)
            log.info(f"  -> Fold {fold + 1} finished.")

        log.info("🚀 Final Fit on Full Data...")
        self.final_preprocessor = SmartPreprocessor(n_clusters=N_CLUSTERS).fit(X_raw_full)
        X_full = self.final_preprocessor.transform(X_raw_full)

        self.trained_models = []
        for name, cls, params in tqdm(self.configs, desc="Final Models"):
            self.trained_models.append(self._fit_single(cls, params, X_full, y_full, weights_full, name))

        mlp_final = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=200, random_state=RANDOM_STATE)
        mlp_final.fit(X_full, y_full)
        self.trained_models.append(mlp_final)

        self.final_estimator.fit(meta_features, y_full)
        return self

    def predict(self, X_raw, calibrate=True):
        X = self.final_preprocessor.transform(X_raw)
        preds = []
        for m in self.trained_models:
            preds.append(m.predict(X))
        meta = np.column_stack(preds)

        logits = self.final_estimator.predict(meta)
        res = from_logit(logits)
        if calibrate: res = self.bias_corrector.predict(res)
        return res

    def calibrate(self, X_calib, y_calib):
        y_pred_unc = self.predict(X_calib, calibrate=False)
        self.bias_corrector.fit(y_pred_unc, y_calib)


# --- ТОЧКА ВХОДА ---
if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        log.error(f"❌ Файл {DATA_FILE} не найден! Положи его рядом со скриптом.")
        sys.exit(1)

    log.info("--- 1. Чтение данных ---")
    df = pd.read_csv(DATA_FILE, decimal=',', sep=';', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)

    X, y = df.drop(columns=['target']), df['target']
    X_main, X_test, y_main, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)
    X_train, X_calib, y_train, y_calib = train_test_split(X_main, y_main, test_size=0.05, random_state=RANDOM_STATE)

    y_train_logit = to_logit(y_train)
    w_train = calculate_weights(y_train)

    log.info("--- 2. Optuna Search ---")
    pre_tmp = SmartPreprocessor(n_clusters=N_CLUSTERS).fit(X_train)
    if len(X_train) > OPTUNA_SAMPLE:
        idx = np.random.choice(X_train.index, size=OPTUNA_SAMPLE, replace=False)
        X_opt_raw, y_opt, w_opt = X_train.loc[idx], y_train_logit.loc[idx], w_train.loc[idx]
    else:
        X_opt_raw, y_opt, w_opt = X_train, y_train_logit, w_train
    X_opt = pre_tmp.transform(X_opt_raw)

    p_xgb_q = run_optuna_local('optuna_xgb_q', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'xgb', 'q'), N_TRIALS_FAST)
    p_xgb_m = run_optuna_local('optuna_xgb_m', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'xgb', 'm'), N_TRIALS_FAST)

    p_lgbm_q = run_optuna_local('optuna_lgbm_q', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'lgbm', 'q'),
                                N_TRIALS_FAST)
    p_lgbm_m = run_optuna_local('optuna_lgbm_m', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'lgbm', 'm'),
                                N_TRIALS_FAST)

    p_cat_q = run_optuna_local('optuna_cat_q', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'cat', 'q'), N_TRIALS_SLOW)
    p_cat_m = run_optuna_local('optuna_cat_m', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'cat', 'm'), N_TRIALS_SLOW)

    log.info("\n--- 3. Сборка HybridModelGod ---")
    model = HybridModelGod(p_lgbm_q, p_lgbm_m, p_xgb_q, p_xgb_m, p_cat_q, p_cat_m)
    model.fit(X_train, y_train_logit, w_train)

    log.info("--- 4. Калибровка ---")
    model.calibrate(X_calib, y_calib)

    log.info("--- 5. Финальная проверка ---")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    log.info(f"📊 RESULT RMSE: {rmse:.5f}")
    log.info(f"📊 RESULT R2:   {r2:.5f}")

    save_path = os.path.join(PROJECT_DIR, FINAL_MODEL_NAME)
    joblib.dump(model, save_path)
    log.info(f"💾 Модель сохранена: {save_path}")