# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

# --- КОНФИГУРАЦИЯ ---
MODEL_FILE = 'final_model_god_v8_gpu_only.joblib'
DATA_FILE = 'unique_rows.csv'
EPSILON = 1e-4
RANDOM_STATE = 42

# --- ПАТЧ ДЛЯ CPU (ВАЖНО ДЛЯ МОДЕЛЕЙ С GPU) ---
if not torch.cuda.is_available():
    import torch.serialization

    _original_load = torch.load
    torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'map_location': torch.device('cpu')})
    print("⚠️ Режим CPU: патч десериализации активирован.")


# --- ОПРЕДЕЛЕНИЯ КЛАССОВ (ДОЛЖНЫ ТОЧНО СОВПАДАТЬ С ОРИГИНАЛОМ) ---

class ChampionWrapper:
    def __init__(self, model=None, metadata=None):
        self.model = model
        self.metadata = metadata


class TorchMLP(BaseEstimator):
    def __init__(self, **kwargs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

    def predict(self, X):
        if self.model is None: return np.zeros(len(X))
        self.model.eval()
        dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_t = torch.tensor(np.array(X), dtype=torch.float32).to(dev)
        with torch.no_grad(): return self.model(X_t).cpu().numpy().flatten()


class SmartPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=15):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
        self.n_clusters = n_clusters

    def transform(self, X_raw):
        # Восстанавливаем полную математику признаков
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)

        # Базовые статистики по строке
        df['row_mean'] = df.mean(axis=1)
        df['row_std'] = df.std(axis=1)

        # Кластерные дистанции
        dists = self.kmeans.transform(X_scaled)
        for i in range(self.n_clusters):
            df[f'dist_clust_{i}'] = dists[:, i]

        # ID кластера
        df['cluster_id'] = self.kmeans.predict(X_scaled)

        # Математические комбинации (как в исходном коде обучения)
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

    def predict(self, y_p):
        return self.iso.predict(y_p)


class HybridModelGod:
    def __init__(self, **kwargs):
        self.bias_corrector = IsotonicCorrector()
        self.final_preprocessor = None
        self.trained_models = []
        self.configs = []
        self.final_estimator = None

    def predict(self, X_raw, calibrate=True):
        X = self.final_preprocessor.transform(X_raw)
        preds = [m.predict(X) for m in self.trained_models]
        meta = np.column_stack(preds)
        logits = self.final_estimator.predict(meta)
        res = 1 / (1 + np.exp(-logits))
        return self.bias_corrector.predict(res) if calibrate else res


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def check_bias(y_true, y_pred):
    diff = np.mean(y_pred - y_true)
    if diff > 0.0005: return f"Завышает (+{diff:.5f})"
    if diff < -0.0005: return f"Занижает ({diff:.5f})"
    return "Нейтрально"


# --- MAIN ---

def main():
    if not os.path.exists(MODEL_FILE): return print(f"❌ Файл {MODEL_FILE} не найден.")

    print(f"⏳ Загрузка модели...")
    loaded = joblib.load(MODEL_FILE)
    model = loaded.model if isinstance(loaded, ChampionWrapper) else loaded

    # Исправляем девайс для нейронок
    for m in model.trained_models:
        if isinstance(m, TorchMLP):
            m.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if m.model: m.model.to(m.device)

    print(f"⏳ Загрузка данных {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE, decimal=',', sep=';', header=None, skiprows=1)
    X_test = df[[1, 2, 3, 4]]
    X_test.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    y_test = pd.to_numeric(df[5], errors='coerce')
    mask = ~y_test.isna();
    X_test, y_test = X_test[mask], y_test[mask]

    # Трансформация данных (теперь с полным набором фич)
    X_proc = model.final_preprocessor.transform(X_test).astype(np.float32)

    results = {}
    print("🚀 Расчет предсказаний для всех моделей...")

    # 1. Базовые модели из конфига
    for i, (name, _, _) in enumerate(model.configs):
        p_logit = model.trained_models[i].predict(X_proc)
        results[name] = 1 / (1 + np.exp(-p_logit))

    # 2. Нейронка (последняя в списке)
    mlp_logit = model.trained_models[-1].predict(X_proc)
    results['Torch_MLP'] = 1 / (1 + np.exp(-mlp_logit))

    # 3. Финальный ансамбль
    results['FINAL_STACKING'] = model.predict(X_test)

    summary_rows = []

    for name, preds in results.items():
        # Метрики
        rmse_full = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # Зона 0.8 - 1.0
        mask_h = y_test >= 0.8
        y_h, p_h = y_test[mask_h], preds[mask_h]
        rmse_h = np.sqrt(mean_squared_error(y_h, p_h)) if len(y_h) > 0 else 0
        bias_h = check_bias(y_h, p_h) if len(y_h) > 0 else "N/A"

        # Сохранение TXT для каждой модели
        with open(f"report_{name}.txt", "w", encoding="utf-8") as f:
            f.write(f"MODEL: {name}\n")
            f.write(f"RMSE Global: {rmse_full:.6f}\n")
            f.write(f"R2 Score:    {r2:.6f}\n")
            f.write(f"RMSE (0.8-1): {rmse_h:.6f}\n")
            f.write(f"Bias (0.8-1): {bias_h}\n")

        summary_rows.append([name, rmse_full, r2, rmse_h, bias_h])

        # ГРАФИКИ (3 в 1)
        fig, axes = plt.subplots(1, 3, figsize=(21, 6))

        # Полный график
        axes[0].hexbin(y_test, preds, gridsize=30, cmap='viridis', mincnt=1)
        axes[0].plot([0, 1], [0, 1], 'r--', alpha=0.5)
        axes[0].set_title(f"Full Range\nRMSE: {rmse_full:.4f}")

        # Зум 0.8 - 1.0
        axes[1].hexbin(y_test, preds, gridsize=20, cmap='magma', mincnt=1)
        axes[1].plot([0.8, 1], [0.8, 1], 'r--', lw=2)
        axes[1].set_xlim(0.78, 1.01);
        axes[1].set_ylim(0.78, 1.01)
        axes[1].set_title(f"Zoom 0.8-1.0\nBias: {bias_h}")

        # Распределение (Байес/KDE)
        sns.kdeplot(y_test, label='Real', ax=axes[2], fill=True, alpha=0.2, color="blue")
        sns.kdeplot(preds, label='Pred', ax=axes[2], fill=True, alpha=0.2, color="orange")
        axes[2].set_title("Density Distribution")
        axes[2].legend()

        plt.suptitle(f"Analysis: {name}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(f"plots_{name}.png")
        plt.close()

    # Общий текстовый отчет
    df_sum = pd.DataFrame(summary_rows, columns=['Model', 'RMSE_Glob', 'R2', 'RMSE_0.8_1', 'Bias_0.8_1'])
    df_sum.to_csv("ALL_MODELS_SUMMARY.txt", sep="\t", index=False)

    print("\n✅ Успешно завершено!")
    print(f"📊 Сводный отчет в 'ALL_MODELS_SUMMARY.txt'")
    print(df_sum.sort_values('RMSE_0.8_1').to_string(index=False))


if __name__ == "__main__":
    main()