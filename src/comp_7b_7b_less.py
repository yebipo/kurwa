# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# --- ЗАГЛУШКИ КЛАССОВ (ОБЯЗАТЕЛЬНО ДЛЯ ЗАГРУЗКИ) ---
class SmartPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=15):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.n_clusters = n_clusters

    def fit(self, X, y=None): return self

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

    def fit(self, y_pred, y_true): return self

    def predict(self, y_pred): return self.iso.predict(y_pred)


class AverageBlender(BaseEstimator, RegressorMixin):
    def fit(self, X, y): return self

    def predict(self, X): return np.mean(X, axis=1)


class HybridModelGod:
    def __init__(self, p_lgbm_q, p_lgbm_m, p_xgb_q, p_xgb_m, p_cat_q, p_cat_m):
        self.bias_corrector = IsotonicCorrector()
        self.final_preprocessor = None
        self.final_estimator = Ridge(alpha=1.0, random_state=42)
        self.trained_models = []

    def predict(self, X_raw, calibrate=True):
        X = self.final_preprocessor.transform(X_raw)
        preds = [m.predict(X) for m in self.trained_models]
        meta = np.column_stack(preds)
        logits = self.final_estimator.predict(meta)
        res = 1 / (1 + np.exp(-logits))
        res = np.clip(res, 0, 1)
        if calibrate: res = self.bias_corrector.predict(res)
        return res


# --- КОНФИГ ---
DATA_FILE = 'million.csv'
OLD_MODEL = 'final_model_god_v4.joblib'
NEW_MODEL = 'final_model_god_v4_OPTIMIZED.joblib'
IMG_NAME = 'comparison_plot.png'


def evaluate(y_true, y_pred, name):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Фильтрация для диапазона 0.8 - 1.0
    mask_high = (y_true >= 0.8) & (y_true <= 1.0)
    if mask_high.sum() > 0:
        rmse_high = np.sqrt(mean_squared_error(y_true[mask_high], y_pred[mask_high]))
        r2_high = r2_score(y_true[mask_high], y_pred[mask_high])
    else:
        rmse_high, r2_high = 0.0, 0.0

    print(f"🔹 {name}:")
    print(f"   Total RMSE: {rmse:.5f} | R2: {r2:.5f}")
    print(f"   [0.8-1.0] RMSE: {rmse_high:.5f} | R2: {r2_high:.5f}")
    print(f"   Count in range: {mask_high.sum()}")
    return rmse, rmse_high


def main():
    print("⏳ Читаем данные...")
    try:
        df = pd.read_csv(DATA_FILE, decimal=',', sep=';', header=None, skiprows=1,
                         usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
        # --- ИСПРАВЛЕНИЕ: Имена должны точно совпадать с теми, на которых учили ---
        df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    except:
        df = pd.read_csv(DATA_FILE, decimal=',', sep=';')

    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)
    X, y = df.drop(columns=['target']), df['target']

    # Сплит
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ограничим размер для графика
    if len(X_val) > 10000:
        indices = np.random.choice(len(X_val), 10000, replace=False)
        X_plot = X_val.iloc[indices]
        y_plot = y_val.iloc[indices]
    else:
        X_plot, y_plot = X_val, y_val

    print(f"⚔️ Сравниваем модели на {len(X_val)} примерах...")

    # 1. Загрузка
    try:
        m_old = joblib.load(OLD_MODEL)
        m_new = joblib.load(NEW_MODEL)
    except FileNotFoundError:
        print("❌ Не найдены файлы моделей (.joblib)")
        return

    # 2. Предикт
    print("🔮 Predict Old...")
    p_old = m_old.predict(X_val)
    print("🔮 Predict New...")
    p_new = m_new.predict(X_val)

    # 3. Метрики
    evaluate(y_val, p_old, "OLD Model (Ridge)")
    evaluate(y_val, p_new, "NEW Model (Avg + Calib)")

    # 4. Графики
    print("🎨 Рисуем графики...")
    p_old_viz = m_old.predict(X_plot)
    p_new_viz = m_new.predict(X_plot)

    plt.figure(figsize=(14, 6))

    # Subplot 1: Scatter
    plt.subplot(1, 2, 1)
    # Рисуем Old снизу красным, New сверху синим (поменьше прозрачность)
    plt.scatter(y_plot, p_old_viz, alpha=0.2, s=5, label='Old', color='red')
    plt.scatter(y_plot, p_new_viz, alpha=0.2, s=5, label='New', color='blue')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('True Value')
    plt.ylabel('Predicted')
    plt.title('True vs Predicted (Sample 10k)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Errors on 0.8-1.0
    mask_high = (y_plot >= 0.8)
    if mask_high.sum() > 0:
        err_old = p_old_viz[mask_high] - y_plot[mask_high]
        err_new = p_new_viz[mask_high] - y_plot[mask_high]

        plt.subplot(1, 2, 2)
        plt.hist(err_old, bins=50, alpha=0.4, label=f'Old (std={np.std(err_old):.4f})', color='red', density=True)
        plt.hist(err_new, bins=50, alpha=0.4, label=f'New (std={np.std(err_new):.4f})', color='blue', density=True)
        plt.axvline(0, color='k', linestyle='--')
        plt.title('Error Distribution for Target >= 0.8')
        plt.xlabel('Error (Pred - True)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    else:
        print("⚠️ Нет данных в диапазоне 0.8-1.0 для построения гистограммы.")

    plt.tight_layout()
    plt.savefig(IMG_NAME, dpi=150)
    print(f"📸 График сохранен: {IMG_NAME}")


if __name__ == "__main__":
    main()