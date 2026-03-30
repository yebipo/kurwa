import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_squared_error
import warnings

warnings.filterwarnings('ignore')

# --- 1. ОПРЕДЕЛЕНИЯ КЛАССОВ (ОСТАВЛЯЕМ ДЛЯ ЗАГРУЗКИ) ---
EPSILON = 1e-4


class SmartPreprocessor(BaseEstimator):
    def __init__(self, n_clusters=15):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.n_clusters = n_clusters

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


class ChampionWrapper:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, X_raw):
        X_trans = self.preprocessor.transform(X_raw)
        preds = self.model.predict(X_trans)
        if preds.min() < -0.5 or preds.max() > 1.5:
            preds = 1 / (1 + np.exp(-preds))
        return np.clip(preds, 0, 1)


class CorrectedChampionWrapper:
    def __init__(self, base_model, corrector_model):
        self.base_model = base_model
        self.corrector_model = corrector_model

    def predict(self, X_raw):
        # Метод с коррекцией (сейчас мы его обойдем)
        base_preds = self.base_model.predict(X_raw)
        correction = self.corrector_model.predict(base_preds.reshape(-1, 1))
        return np.clip(base_preds + correction, 0, 1)


# --- 2. ЛОГИКА ТЕСТА БЕЗ КОРРЕКТОРА ---
FILE_ADEQUATE = 'output_adequate.csv'
FILE_MILLION = 'million.csv'
FILE_MODEL = 'champion_corrected.joblib'
FILE_OUTPUT = 'pure_base_results.csv'


def run_pure_test():
    print("📂 Шаг 1: Загрузка данных...")

    def load_user_style(path):
        df = pd.read_csv(path, sep=';', decimal=',', header=None, skiprows=1,
                         usecols=[1, 2, 3, 4, 5], engine='c')
        df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.'), errors='coerce')
        df.dropna(subset=['target'], inplace=True)
        return df

    df_adeq = load_user_style(FILE_ADEQUATE)
    df_mill = load_user_style(FILE_MILLION)

    print("✂️ Вычитание дублей...")
    df_diff = df_adeq.merge(df_mill, how='left', indicator=True)
    df_final = df_diff[df_diff['_merge'] == 'left_only'].drop(columns=['_merge'])

    if df_final.empty:
        print("❌ Данных нет.")
        return

    print(f"🔄 Загрузка модели {FILE_MODEL}...")
    original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: original_torch_load(*args, **kwargs, map_location='cpu')
    try:
        full_model = joblib.load(FILE_MODEL)
        # ИСПОЛЬЗУЕМ ТОЛЬКО БАЗУ
        model = full_model.base_model
    finally:
        torch.load = original_torch_load

    X_raw = df_final[['feature_1', 'feature_2', 'feature_3', 'feature_4']]
    y_real = df_final['target'].values

    print("🚀 Запуск ПРЯМОГО предикта (Base Model only)...")
    try:
        y_pred = model.predict(X_raw)

        # Сохранение (предикт на 5-й позиции)
        df_final.insert(4, 'pure_prediction', y_pred)
        df_final.to_csv(FILE_OUTPUT, sep=';', decimal=',', index=False, encoding='utf-8-sig')

        # Метрики
        r2 = r2_score(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))

        print("\n" + "═" * 40)
        print(f"📊 РЕЗУЛЬТАТЫ БЕЗ КОРРЕКТОРА:")
        print(f"   R2 Score: {r2:.6f}")
        print(f"   RMSE:     {rmse:.6f}")
        print("═" * 40)

        # График диагностики
        plt.figure(figsize=(10, 6))
        plt.scatter(y_real, y_pred, alpha=0.15, s=1, color='green')
        plt.plot([0, 1], [0, 1], 'r--', lw=2)
        plt.xlabel('Реальный таргет')
        plt.ylabel('Прогноз (Pure Base)')
        plt.title(f'Base Model Test\nR2: {r2:.4f}')
        plt.grid(True, alpha=0.2)
        plt.savefig('pure_base_plot.png')
        print("📈 График сохранен: 'pure_base_plot.png'")

    except Exception as e:
        print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    run_pure_test()