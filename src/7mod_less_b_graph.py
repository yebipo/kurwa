# -*- coding: utf-8 -*-
import os
import sys
import logging
import gc
import pandas as pd
import numpy as np
import joblib
import optuna
import time
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- НАСТРОЙКИ ---
warnings.filterwarnings('ignore')

DATA_FILE = 'million.csv'
MODEL_FILE = 'final_model_god_v8_gpu_only.joblib'
CHUNK_SIZE = 100000
RANDOM_STATE = 42
LOG_FILE = 'cpu_inference_log.txt'

QUANTILE_ALPHA_LOW = 0.1
QUANTILE_ALPHA_MID = 0.5
QUANTILE_ALPHA_HIGH = 0.9
N_CLUSTERS = 15

# ПРИНУДИТЕЛЬНО CPU
DEVICE = torch.device('cpu')


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
log.info(f"🖥️ COMPUTE DEVICE: {DEVICE} (Forced CPU Mode ❄️)")

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


# --- ФУНКЦИЯ КОРРЕКЦИИ (FIX V2 - SURGICAL) ---
def fix_high_range_bias(y_pred):
    """
    Исправленная версия v2.
    Защищает зону 0.8-0.89 от ложного срабатывания.
    Тянет вверх только уверенные предикты (>0.89).
    """
    y_fixed = np.array(y_pred).copy()

    # Порог поднят, чтобы не цеплять "шумные 0.8"
    threshold = 0.89

    # Фактор увеличен, чтобы успеть дотянуть 0.94 до 1.0 на коротком отрезке
    boost_factor = 1.5

    mask = y_fixed > threshold
    if mask.sum() > 0:
        y_fixed[mask] = y_fixed[mask] + (y_fixed[mask] - threshold) * boost_factor

    return np.clip(y_fixed, 0.0, 1.0)


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
        self.device = torch.device('cpu')
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
        if self.model is None:
            return np.zeros(X.shape[0])

        if next(self.model.parameters()).device.type != 'cpu':
            self.model.to('cpu')
            self.device = torch.device('cpu')

        self.model.eval()
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to('cpu')

        with torch.no_grad():
            preds = self.model(X_tensor).cpu().numpy().flatten()
        return preds


# --- Helper for MLP params ---
def rebuild_mlp_params(params):
    p = params.copy()
    n_layers = p.pop('n_layers', 2)
    layers = []
    if 'n_units_l0' in p:
        layers.append(p.pop('n_units_l0'))
    for i in range(1, n_layers):
        key = f'n_units_l{i}'
        if key in p:
            layers.append(p.pop(key))
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
              'tree_method': 'hist', 'verbosity': 0, 'n_jobs': -1}),
            ('xgb_m', xgb.XGBRegressor,
             {**adj(p_xgb_m), 'objective': 'reg:squarederror',
              'tree_method': 'hist', 'verbosity': 0, 'n_jobs': -1}),
            ('cat_q_mid', cb.CatBoostRegressor,
             {**adj(p_cat_q, 'iterations'), 'loss_function': f'Quantile:alpha={QUANTILE_ALPHA_MID}',
              'task_type': 'CPU', 'thread_count': -1, 'verbose': 0}),
            ('cat_m', cb.CatBoostRegressor,
             {**adj(p_cat_m, 'iterations'), 'loss_function': 'RMSE',
              'task_type': 'CPU', 'thread_count': -1, 'verbose': 0}),
            ('lgbm_q_low', lgb.LGBMRegressor,
             {**adj(p_lgbm_q), 'objective': 'quantile', 'alpha': QUANTILE_ALPHA_LOW, 'device': 'cpu', 'n_jobs': -1,
              'verbose': -1}),
            ('xgb_q_low', xgb.XGBRegressor,
             {**adj(p_xgb_q), 'objective': 'reg:quantileerror', 'quantile_alpha': QUANTILE_ALPHA_LOW,
              'tree_method': 'hist', 'verbosity': 0, 'n_jobs': -1}),
            ('lgbm_q_high', lgb.LGBMRegressor,
             {**adj(p_lgbm_q), 'objective': 'quantile', 'alpha': QUANTILE_ALPHA_HIGH, 'device': 'cpu', 'n_jobs': -1,
              'verbose': -1}),
            ('xgb_q_high', xgb.XGBRegressor,
             {**adj(p_xgb_q), 'objective': 'reg:quantileerror', 'quantile_alpha': QUANTILE_ALPHA_HIGH,
              'tree_method': 'hist', 'verbosity': 0, 'n_jobs': -1}),
        ]

        self.final_estimator = cb.CatBoostRegressor(
            iterations=300, depth=3, learning_rate=0.08,
            loss_function='RMSE', task_type='CPU', thread_count=-1, verbose=0, random_state=RANDOM_STATE
        )
        self.trained_models = []

        clean_mlp_params = rebuild_mlp_params(p_mlp)
        clean_mlp_params['device'] = torch.device('cpu')
        self.mlp_meta_model = TorchMLP(**clean_mlp_params)

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


# --- ЛЕЧЕНИЕ МОДЕЛИ ---
def fix_loaded_model_for_cpu(wrapper):
    log.info("🔧 Fixing loaded model parameters for CPU usage...")
    for i, model in enumerate(wrapper.trained_models):
        if isinstance(model, xgb.XGBRegressor) or isinstance(model, xgb.XGBModel):
            model.set_params(tree_method='hist', device='cpu', n_jobs=-1)
        elif isinstance(model, cb.CatBoostRegressor):
            if hasattr(model, '_init_params'):
                model._init_params['task_type'] = 'CPU'
                model._init_params['devices'] = None
        elif isinstance(model, TorchMLP):
            model.device = torch.device('cpu')
            if model.model:
                model.model.to('cpu')

    if hasattr(wrapper, 'mlp_meta_model'):
        wrapper.mlp_meta_model.device = torch.device('cpu')
        if wrapper.mlp_meta_model.model:
            wrapper.mlp_meta_model.model.to('cpu')

    if hasattr(wrapper, 'final_estimator'):
        fe = wrapper.final_estimator
        if hasattr(fe, '_init_params'):
            fe._init_params['task_type'] = 'CPU'
            fe._init_params['devices'] = None

    log.info("✅ Model sanitized. Ready for CPU inference.")
    return wrapper


# --- MAIN ---
def main():
    print("=== CPU INFERENCE MODE (SURGICAL FIX v2) ===\n")

    if not os.path.exists(MODEL_FILE) and not os.path.exists(os.path.join('project_files', MODEL_FILE)):
        print(f"❌ Модель {MODEL_FILE} не найдена.")
        return

    m_path = MODEL_FILE if os.path.exists(MODEL_FILE) else os.path.join('project_files', MODEL_FILE)
    print(f"📥 Loading Model: {m_path}...")

    # Monkey patch for torch.load
    original_torch_load = torch.load

    def cpu_load(*args, **kwargs):
        kwargs['map_location'] = torch.device('cpu')
        return original_torch_load(*args, **kwargs)

    torch.load = cpu_load

    try:
        wrapper = joblib.load(m_path)
    except Exception as e:
        print(f"CRITICAL ERROR during loading: {e}")
        torch.load = original_torch_load
        return
    finally:
        torch.load = original_torch_load

    wrapper = fix_loaded_model_for_cpu(wrapper)

    if not os.path.exists(DATA_FILE):
        print(f"❌ Файл данных {DATA_FILE} не найден.")
        return

    y_true_all = []
    preds_all = []

    print(f"🚜 Reading & Predicting...")
    total_lines = sum(1 for _ in open(DATA_FILE)) - 1

    reader = pd.read_csv(DATA_FILE, sep=';', decimal=',', header=None, skiprows=1,
                         usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys(),
                         chunksize=CHUNK_SIZE)
    cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']

    with tqdm(total=total_lines, unit="rows") as pbar:
        for chunk in reader:
            chunk.columns = cols if len(chunk.columns) == 5 else chunk.columns
            chunk['target'] = pd.to_numeric(chunk['target'], errors='coerce')
            chunk.dropna(subset=['target'], inplace=True)
            if chunk.empty: continue

            try:
                # 1. Предикт
                raw_preds = wrapper.predict(chunk.drop(columns=['target']), calibrate=True)
                # 2. Фикс (Surgical v2)
                fixed_preds = fix_high_range_bias(raw_preds)

                preds_all.extend(fixed_preds)
                y_true_all.extend(chunk['target'].values)
            except Exception as e:
                print(f"Error in chunk: {e}")

            pbar.update(len(chunk))
            del chunk, raw_preds, fixed_preds
            gc.collect()

    y_real = np.array(y_true_all)
    y_pred = np.array(preds_all)

    print(f"\n📊 Processed: {len(y_real)} points.")

    rmse_total = np.sqrt(mean_squared_error(y_real, y_pred))

    mask_high = (y_real >= 0.8) & (y_real <= 1.0)
    if mask_high.sum() > 0:
        rmse_high = np.sqrt(mean_squared_error(y_real[mask_high], y_pred[mask_high]))
        count_high = mask_high.sum()
    else:
        rmse_high = 0.0
        count_high = 0

    print("=" * 40)
    print(f"🌍 TOTAL RMSE:       {rmse_total:.5f}")
    print(f"🔥 HIGH RANGE RMSE:  {rmse_high:.5f}  (on {count_high} points)")
    print("=" * 40)

    # --- ВИЗУАЛИЗАЦИЯ ---
    plt.figure(figsize=(18, 6))

    # 1. Full Range
    plt.subplot(1, 3, 1)
    plt.hexbin(y_real, y_pred, gridsize=50, cmap='inferno', mincnt=1, bins='log')
    plt.plot([0, 1], [0, 1], 'w--', lw=2)
    plt.title(f"FULL RANGE (RMSE: {rmse_total:.4f})")
    plt.xlabel("Real")
    plt.ylabel("Pred")
    plt.colorbar(label='Log Count')

    # 2. High Range Zoom (0.8 - 1.0)
    plt.subplot(1, 3, 2)
    if count_high > 0:
        plt.hexbin(y_real[mask_high], y_pred[mask_high], gridsize=30, cmap='viridis', mincnt=1)
        plt.plot([0.8, 1], [0.8, 1], 'r--', lw=2)
        plt.title(f"HIGH RANGE ZOOM (RMSE: {rmse_high:.4f})")
        plt.xlabel("Real")
        plt.ylabel("Pred")

    # 3. Error Dist
    plt.subplot(1, 3, 3)
    sns.histplot(y_real - y_pred, bins=100, color='blue', kde=True)
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Error Distribution")
    plt.xlabel("Residual (Real - Pred)")

    plt.tight_layout()
    plt.savefig("cpu_results_surgical.png")
    print("🖼️ Plot saved to cpu_results_surgical.png")
    plt.show()


if __name__ == "__main__":
    main()