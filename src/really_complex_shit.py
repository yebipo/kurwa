# -*- coding: utf-8 -*-
import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
import optuna
import time
import warnings
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

# --- PYTORCH IMPORTS ---
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Игнорируем лишние предупреждения
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- КОНФИГУРАЦИЯ ---
DATA_FILE = 'million.csv'
FINAL_MODEL_NAME = 'final_model_god_v7_turbo.joblib'
LOG_FILE = 'training_log_v7_turbo.txt'

# Параметры обучения [TURBO MODE: ON]
# Мы используем старые базы, но с новыми, более быстрыми лимитами.
N_TRIALS_FAST = 80  # Быстро ищем (было 200). Если в базе уже 17, он сделает еще 23.
N_TRIALS_SLOW = 80  # Кэтбуст умный (было 80)
N_TRIALS_MLP = 100  # Нейронке хватит (было 100)
OPTUNA_SAMPLE = 50000  # Сэмпл меньше для скорости (было 50к)
N_CLUSTERS = 15

QUANTILE_ALPHA_MID = 0.50
QUANTILE_ALPHA_LOW = 0.10
QUANTILE_ALPHA_HIGH = 0.90
RANDOM_STATE = 42


# Настройка Логгера
def setup_logger():
    logger = logging.getLogger('EntropyLogger')
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(fh)
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(ch)
    return logger


log = setup_logger()

# Проверка GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"🖥️ COMPUTE DEVICE: {DEVICE} (Using {'3080 Ti Power 🔥' if 'cuda' in str(DEVICE) else 'CPU ❄️'})")

# --- МАТЕМАТИКА ---
EPSILON = 1e-4


def to_logit(y):
    y_safe = np.clip(y, EPSILON, 1 - EPSILON)
    return np.log(y_safe / (1 - y_safe))


def from_logit(z):
    res = 1 / (1 + np.exp(-z))
    return np.clip(res, 0, 1)


def calculate_weights(y):
    y_safe = np.clip(y, 0, 1)
    weights = np.ones_like(y_safe)
    threshold = 0.75
    max_weight = 3.0
    mask = y_safe > threshold
    if mask.sum() > 0:
        weights[mask] = 1.0 + (max_weight - 1.0) * ((y_safe[mask] - threshold) / (1 - threshold))
    return weights


def pinball_loss(y_true, y_pred, alpha):
    residual = y_true - y_pred
    return np.mean(np.maximum(alpha * residual, (alpha - 1) * residual))


# --- CUSTOM PYTORCH MLP ---
class TorchMLP(BaseEstimator):
    def __init__(self, hidden_layer_sizes=(128, 64), activation='relu',
                 alpha=0.0001, learning_rate_init=0.001, max_iter=200,
                 batch_size=2048, device=DEVICE, random_state=42):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.model = None

    def fit(self, X, y):
        torch.manual_seed(self.random_state)
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        y_vals = y.values if hasattr(y, 'values') else y
        y_tensor = torch.tensor(np.array(y_vals), dtype=torch.float32).view(-1, 1).to(self.device)

        input_dim = X.shape[1]
        layers = []
        in_features = input_dim

        for units in self.hidden_layer_sizes:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.BatchNorm1d(units))
            layers.append(nn.ReLU() if self.activation == 'relu' else nn.Tanh())
            in_features = units
        layers.append(nn.Linear(in_features, 1))

        self.model = nn.Sequential(*layers).to(self.device)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        criterion = nn.MSELoss()

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model.train()
        for epoch in range(self.max_iter):
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(self.device)
        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
        return preds


# --- [FIX] helper to fix optuna params for MLP ---
def rebuild_mlp_params(params):
    """Превращает плоские параметры Optuna обратно в структуру для TorchMLP"""
    p = params.copy()
    n_layers = p.pop('n_layers', 2)  # Забираем и удаляем служебный ключ

    layers = []
    # Собираем слои
    if 'n_units_l0' in p:
        layers.append(p.pop('n_units_l0'))

    for i in range(1, n_layers):
        key = f'n_units_l{i}'
        if key in p:
            layers.append(p.pop(key))

    # Все, что осталось в p (alpha, lr и т.д.) - это аргументы конструктора
    # Добавляем собранные слои
    p['hidden_layer_sizes'] = tuple(layers)
    return p


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

        df['f1_f2_sum'] = df['feature_1'] + df['feature_2']
        df['f1_f2_diff'] = df['feature_1'] - df['feature_2']
        df['f3_f4_sum'] = df['feature_3'] + df['feature_4']
        df['group1_x_group2'] = df['f1_f2_sum'] * df['f3_f4_sum']
        df['f1_over_f3'] = df['feature_1'] / (df['feature_3'] + EPSILON)
        df['f2_over_f4'] = df['feature_2'] / (df['feature_4'] + EPSILON)
        df['row_median'] = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']].median(axis=1)

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
            'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10, log=True),
            'verbosity': 0, 'tree_method': 'hist', 'device': 'cuda'
        }
    elif model_name == 'lgbm':
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 600, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10, log=True),
            'device': 'cpu', 'n_jobs': -1, 'verbose': -1
        }
    elif model_name == 'cat':
        p = {
            'iterations': trial.suggest_int('iterations', 600, 1500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1),
            'depth': trial.suggest_int('depth', 4, 9),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
            'task_type': 'GPU', 'devices': '0', 'verbose': 0
        }
    elif model_name == 'mlp':
        n_layers = trial.suggest_int('n_layers', 2, 5)
        layers = []
        layers.append(trial.suggest_int('n_units_l0', 64, 512))
        for i in range(1, n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 32, 256))

        p = {
            'hidden_layer_sizes': tuple(layers),
            'activation': 'relu',
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-2, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 0.01, log=True),
            'max_iter': 100,
            'batch_size': 4096
        }
    return p


def obj_generic(trial, X, y, w, model_name, obj_type):
    X = X.astype(np.float32)
    X_t, X_v, y_t, y_v, w_t, _ = train_test_split(X, y, w, test_size=0.2, random_state=42)
    p = get_params(trial, model_name)
    y_v_orig = from_logit(y_v)

    try:
        model = None
        if model_name == 'xgb':
            if obj_type == 'q':
                p.update({'objective': 'reg:quantileerror', 'quantile_alpha': QUANTILE_ALPHA_MID})
            else:
                p.update({'objective': 'reg:squarederror'})
            model = xgb.XGBRegressor(**p)
            model.fit(X_t, y_t, sample_weight=w_t)

        elif model_name == 'lgbm':
            if obj_type == 'q':
                p.update({'objective': 'quantile', 'alpha': QUANTILE_ALPHA_MID})
            else:
                p.update({'objective': 'regression', 'metric': 'rmse'})
            model = lgb.LGBMRegressor(**p)
            model.fit(X_t, y_t, sample_weight=w_t)

        elif model_name == 'cat':
            if obj_type == 'q':
                p.update({'loss_function': f'Quantile:alpha={QUANTILE_ALPHA_MID}'})
            else:
                p.update({'loss_function': 'RMSE'})
            model = cb.CatBoostRegressor(**p)
            model.fit(X_t, y_t, sample_weight=w_t)

        elif model_name == 'mlp':
            model = TorchMLP(**p)
            model.fit(X_t, y_t)

        preds_logit = model.predict(X_v)

        if obj_type == 'q':
            loss = pinball_loss(y_v, preds_logit, QUANTILE_ALPHA_MID)
        else:
            preds_orig = from_logit(preds_logit)
            rmse_all = np.sqrt(mean_squared_error(y_v_orig, preds_orig))
            loss = rmse_all

            mask_real_high = y_v_orig > 0.8
            mask_false_high = (preds_orig > 0.8) & (y_v_orig < 0.75)

            if mask_real_high.sum() > 5:
                rmse_high = np.sqrt(mean_squared_error(y_v_orig[mask_real_high], preds_orig[mask_real_high]))
                loss = 0.5 * rmse_all + 0.5 * rmse_high

            if mask_false_high.sum() > 5:
                rmse_false = np.sqrt(mean_squared_error(y_v_orig[mask_false_high], preds_orig[mask_false_high]))
                loss += 0.3 * rmse_false

    except Exception as e:
        return 9999.0
    return loss


def run_optuna_local(study_name, func, n_trials):
    db_path = f"sqlite:///{study_name}.db"
    study = optuna.create_study(study_name=study_name, storage=db_path, load_if_exists=True, direction="minimize")

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    left = n_trials - completed

    if left > 0:
        log.info(f"🔎 {study_name}: Запуск {left} итераций (GPU Mode, RESUMING OLD DB)...")
        with tqdm(total=left) as pbar:
            study.optimize(func, n_trials=left, n_jobs=1, callbacks=[lambda s, t: pbar.update(1)])
    else:
        log.info(f"✅ {study_name}: Уже готово (использованы старые данные).")
    return study.best_params


# --- ГИБРИДНАЯ МОДЕЛЬ ---
class HybridModelGod:
    def __init__(self, p_lgbm_q, p_lgbm_m, p_xgb_q, p_xgb_m, p_cat_q, p_cat_m, p_mlp):
        self.bias_corrector = IsotonicCorrector()
        self.final_preprocessor = None
        self.SCALE_EST = 1.2
        self.SCALE_LR = 0.85

        def adj(p, k_est='n_estimators', k_lr='learning_rate'):
            pc = p.copy()
            if k_est in pc:
                pc[k_est] = int(pc[k_est] * self.SCALE_EST)
                if k_est == 'iterations': pc[k_est] = min(pc[k_est], 3000)
            if k_lr in pc: pc[k_lr] = pc[k_lr] * self.SCALE_LR
            return pc

        self.configs = [
            ('lgbm_q_mid', lgb.LGBMRegressor,
             {**adj(p_lgbm_q), 'objective': 'quantile', 'alpha': QUANTILE_ALPHA_MID, 'device': 'cpu', 'n_jobs': -1,
              'verbose': -1}),
            ('lgbm_m', lgb.LGBMRegressor,
             {**adj(p_lgbm_m), 'objective': 'regression', 'metric': 'rmse', 'device': 'cpu', 'n_jobs': -1,
              'verbose': -1}),
            ('xgb_q_mid', xgb.XGBRegressor,
             {**adj(p_xgb_q), 'objective': 'reg:quantileerror', 'quantile_alpha': QUANTILE_ALPHA_MID,
              'tree_method': 'hist', 'verbosity': 0}),
            ('xgb_m', xgb.XGBRegressor,
             {**adj(p_xgb_m), 'objective': 'reg:squarederror', 'tree_method': 'hist', 'verbosity': 0}),
            ('cat_q_mid', cb.CatBoostRegressor,
             {**adj(p_cat_q, 'iterations'), 'loss_function': f'Quantile:alpha={QUANTILE_ALPHA_MID}', 'task_type': 'GPU',
              'devices': '0', 'verbose': 0}),
            ('cat_m', cb.CatBoostRegressor,
             {**adj(p_cat_m, 'iterations'), 'loss_function': 'RMSE', 'task_type': 'GPU', 'devices': '0', 'verbose': 0}),

            ('lgbm_q_low', lgb.LGBMRegressor,
             {**adj(p_lgbm_q), 'objective': 'quantile', 'alpha': QUANTILE_ALPHA_LOW, 'device': 'cpu', 'n_jobs': -1,
              'verbose': -1}),
            ('xgb_q_low', xgb.XGBRegressor,
             {**adj(p_xgb_q), 'objective': 'reg:quantileerror', 'quantile_alpha': QUANTILE_ALPHA_LOW,
              'tree_method': 'hist', 'verbosity': 0}),

            ('lgbm_q_high', lgb.LGBMRegressor,
             {**adj(p_lgbm_q), 'objective': 'quantile', 'alpha': QUANTILE_ALPHA_HIGH, 'device': 'cpu', 'n_jobs': -1,
              'verbose': -1}),
            ('xgb_q_high', xgb.XGBRegressor,
             {**adj(p_xgb_q), 'objective': 'reg:quantileerror', 'quantile_alpha': QUANTILE_ALPHA_HIGH,
              'tree_method': 'hist', 'verbosity': 0}),
        ]

        self.final_estimator = cb.CatBoostRegressor(
            iterations=300, depth=3, learning_rate=0.08,
            loss_function='RMSE', task_type='GPU', devices='0', verbose=0, random_state=RANDOM_STATE
        )
        self.trained_models = []

        # [FIX] Используем функцию rebuild для очистки параметров от Optuna
        clean_mlp_params = rebuild_mlp_params(p_mlp)
        self.mlp_meta_model = TorchMLP(**clean_mlp_params)

    def _fit_single(self, cls, params, X, y, w, name):
        run_params = params.copy()
        if 'xgb' in name: run_params['device'] = 'cuda'
        if 'cat' in name: run_params['task_type'] = 'GPU'

        m = cls(**run_params)
        m.fit(X, y, sample_weight=w)
        return m

    def fit(self, X_raw_full, y_full, weights_full):
        kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        meta_features = np.zeros((X_raw_full.shape[0], len(self.configs) + 1))

        X_raw_full_float = X_raw_full.astype(np.float32)

        log.info("🚀 Training Stacking (GPU Accelerated CV Phase)...")
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw_full)):
            X_tr_raw = X_raw_full_float.iloc[train_idx]
            X_val_raw = X_raw_full_float.iloc[val_idx]
            y_tr = y_full.iloc[train_idx]
            w_tr = weights_full.iloc[train_idx]

            prep = SmartPreprocessor(n_clusters=N_CLUSTERS).fit(X_tr_raw)
            X_tr = prep.transform(X_tr_raw).astype(np.float32)
            X_val = prep.transform(X_val_raw).astype(np.float32)

            col = 0
            for name, cls, params in self.configs:
                m = self._fit_single(cls, params, X_tr, y_tr, w_tr, name)
                meta_features[val_idx, col] = m.predict(X_val)
                col += 1

            # Обучение Torch MLP
            mlp = TorchMLP(**self.mlp_meta_model.__dict__)
            mlp.model = None
            mlp.fit(X_tr, y_tr)
            meta_features[val_idx, col] = mlp.predict(X_val)
            log.info(f"  -> Fold {fold + 1} finished.")

        log.info("🚀 Final Fit on Full Data...")
        self.final_preprocessor = SmartPreprocessor(n_clusters=N_CLUSTERS).fit(X_raw_full_float)
        X_full = self.final_preprocessor.transform(X_raw_full_float).astype(np.float32)

        self.trained_models = []
        for name, cls, params in tqdm(self.configs, desc="Final Level-0 Models"):
            self.trained_models.append(self._fit_single(cls, params, X_full, y_full, weights_full, name))

        self.mlp_meta_model.fit(X_full, y_full)
        self.trained_models.append(self.mlp_meta_model)

        self.final_estimator.fit(meta_features, y_full, sample_weight=weights_full)
        return self

    def predict(self, X_raw, calibrate=True):
        X = self.final_preprocessor.transform(X_raw.astype(np.float32)).astype(np.float32)
        preds = []
        for m in self.trained_models:
            preds.append(m.predict(X))
        meta = np.column_stack(preds)

        logits = self.final_estimator.predict(meta)
        res = from_logit(logits)
        if calibrate: res = self.bias_corrector.predict(res)
        return np.clip(res, 0.0, 1.0)

    def calibrate(self, X_calib, y_calib):
        y_pred_unc = self.predict(X_calib, calibrate=False)
        self.bias_corrector.fit(y_pred_unc, y_calib)


if __name__ == '__main__':
    if not os.path.exists(DATA_FILE):
        log.error(f"❌ Файл {DATA_FILE} не найден в текущей папке!")
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

    # [FIX] Оборачиваем в Series
    y_train_logit = pd.Series(to_logit(y_train), index=y_train.index)
    w_train = pd.Series(calculate_weights(y_train), index=y_train.index)

    log.info("--- 2. Optuna Search (GPU Mode - RESUMING) ---")
    pre_tmp = SmartPreprocessor(n_clusters=N_CLUSTERS).fit(X_train)
    if len(X_train) > OPTUNA_SAMPLE:
        idx = np.random.choice(X_train.index, size=OPTUNA_SAMPLE, replace=False)
        X_opt_raw, y_opt, w_opt = X_train.loc[idx], y_train_logit.loc[idx], w_train.loc[idx]
    else:
        X_opt_raw, y_opt, w_opt = X_train, y_train_logit, w_train

    X_opt = pre_tmp.transform(X_opt_raw).astype(np.float32)

    # [FIX] ИМЕНА БАЗ ВОЗВРАЩЕНЫ К v6_gpu, ЧТОБЫ ПОДХВАТИТЬ СТАРЫЙ ПРОГРЕСС
    p_xgb_q = run_optuna_local('optuna_xgb_q_v6_gpu', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'xgb', 'q'),
                               N_TRIALS_FAST)
    p_xgb_m = run_optuna_local('optuna_xgb_m_v6_gpu', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'xgb', 'm'),
                               N_TRIALS_FAST)
    p_lgbm_q = run_optuna_local('optuna_lgbm_q_v6_gpu', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'lgbm', 'q'),
                                N_TRIALS_FAST)
    p_lgbm_m = run_optuna_local('optuna_lgbm_m_v6_gpu', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'lgbm', 'm'),
                                N_TRIALS_FAST)
    p_cat_q = run_optuna_local('optuna_cat_q_v6_gpu', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'cat', 'q'),
                               N_TRIALS_SLOW)
    p_cat_m = run_optuna_local('optuna_cat_m_v6_gpu', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'cat', 'm'),
                               N_TRIALS_SLOW)

    log.info("🔎 Запуск поиска для MLP (PyTorch)...")
    p_mlp = run_optuna_local('optuna_mlp_v6_gpu', lambda t: obj_generic(t, X_opt, y_opt, w_opt, 'mlp', 'm'),
                             N_TRIALS_MLP)

    log.info("\n--- 3. Сборка HybridModelGod (Turbo) ---")
    model = HybridModelGod(p_lgbm_q, p_lgbm_m, p_xgb_q, p_xgb_m, p_cat_q, p_cat_m, p_mlp)
    model.fit(X_train, y_train_logit, w_train)

    log.info("--- 4. Калибровка ---")
    model.calibrate(X_calib, y_calib)

    log.info("--- 5. Финальная проверка ---")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mask_high = y_test > 0.8
    if mask_high.sum() > 0:
        rmse_high = np.sqrt(mean_squared_error(y_test[mask_high], y_pred[mask_high]))
        log.info(f"🎯 HIGH RANGE (0.8-1.0) RMSE: {rmse_high:.5f}")

    log.info(f"📊 GLOBAL RMSE: {rmse:.5f}")

    joblib.dump(model, FINAL_MODEL_NAME)
    log.info(f"💾 Модель сохранена: {FINAL_MODEL_NAME}")