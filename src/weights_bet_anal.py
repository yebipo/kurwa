import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# --- НАСТРОЙКИ ---
MODEL_FILENAME = "production_model_v26.pkl"
OUTPUT_CSV = "FULL_distribution_data.csv"  # Сохраняем всё
N_SAMPLES = 2_000_000
BATCH_SIZE = 100_000

# --- ВЕСА (из вашего анализа) ---
F1_VALS = [1, 2, 4]
F1_PROBS = [0.9481, 0.0447, 0.0072]

F2_VALS = [0, 3]
F2_PROBS = [0.9603, 0.0397]

F3_VALS = [16777216, 536870912, 2097152, 262144, 4194304, 32768, 1048576, 512, 4096, 2048, 524288, 131072, 256, 8192,
           268435456, 128, 1073741824, 16384]
F3_PROBS = [0.139, 0.1334, 0.1096, 0.0873, 0.0852, 0.0722, 0.0697, 0.057, 0.0506, 0.047, 0.0466, 0.0433, 0.0243, 0.0229,
            0.0076, 0.0026, 0.0013, 0.0004]

# Feature 4 (Log-Uniform range based on your data)
F4_MIN = 1000
F4_MAX = 300_000_000


def normalize_probs(probs):
    p = np.array(probs)
    return p / p.sum()


# --- КЛАССЫ (ОБЯЗАТЕЛЬНО) ---
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class GlobalPreprocessor:
    def __init__(self, scaler, kmeans):
        self.scaler = scaler
        self.kmeans = kmeans

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        cluster_labels = self.kmeans.predict(X_scaled)
        X_out = X_log.copy()
        X_out['cluster_id'] = cluster_labels
        return X_out


class ProductionModel:
    def __init__(self, preprocessor, lgbm_p, xgb_p, cat_p):
        self.preprocessor = preprocessor
        self.estimators = [
            ('lgbm', lgb.LGBMRegressor(**lgbm_p)),
            ('xgb', xgb.XGBRegressor(**xgb_p)),
            ('catboost', cb.CatBoostRegressor(**cat_p))
        ]
        self.model = StackingRegressor(estimators=self.estimators, final_estimator=LinearRegression(), cv=3, n_jobs=1)

    def fit(self, X_raw, y): pass

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        return self.model.predict(X_ready)


# --- MAIN ---
def main():
    print(f"=== ГЕНЕРАЦИЯ И АНАЛИЗ ВСЕГО СПЕКТРА (N={N_SAMPLES}) ===")

    if not os.path.exists(MODEL_FILENAME):
        print("❌ Нет файла модели.")
        return
    model = joblib.load(MODEL_FILENAME)

    # 1. ГЕНЕРАЦИЯ
    print("🎲 Генерируем входные данные...")
    data = {
        'feature_1': np.random.choice(F1_VALS, size=N_SAMPLES, p=normalize_probs(F1_PROBS)),
        'feature_2': np.random.choice(F2_VALS, size=N_SAMPLES, p=normalize_probs(F2_PROBS)),
        'feature_3': np.random.choice(F3_VALS, size=N_SAMPLES, p=normalize_probs(F3_PROBS)),
        'feature_4': np.exp(np.random.uniform(np.log(F4_MIN), np.log(F4_MAX), N_SAMPLES))
    }
    df = pd.DataFrame(data).round(0)
    df = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']]

    # 2. ПРЕДСКАЗАНИЕ
    print("🧠 Считаем предсказания (без фильтрации)...")
    predictions = []
    for i in tqdm(range(0, N_SAMPLES, BATCH_SIZE)):
        batch = df.iloc[i: i + BATCH_SIZE]
        preds = model.predict(batch)
        predictions.extend(preds)

    df['entropy'] = np.array(predictions)

    # 3. СТАТИСТИКА
    print("\n" + "=" * 40)
    print("📊 СТАТИСТИКА ПРЕДСКАЗАНИЙ:")
    print("=" * 40)
    print(df['entropy'].describe().apply(lambda x: format(x, 'f')))

    max_val = df['entropy'].max()
    mean_val = df['entropy'].mean()

    print(f"\n🔥 МАКСИМУМ, КОТОРЫЙ УДАЛОСЬ НАЙТИ: {max_val:.6f}")
    print(f"📉 СРЕДНЕЕ ЗНАЧЕНИЕ: {mean_val:.6f}")

    # 4. СОХРАНЕНИЕ
    print(f"💾 Сохраняем всё в {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, sep=';', decimal=',', index=False)

    # 5. ОТРИСОВКА (ОДИН БОЛЬШОЙ ГРАФИК)
    print("🎨 Рисуем распределение...")
    plt.figure(figsize=(12, 8))

    # Основная гистограмма
    sns.histplot(df['entropy'], bins=100, kde=True, color='purple', line_kws={'linewidth': 2})

    # Линии статистики
    plt.axvline(mean_val, color='blue', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(max_val, color='red', linestyle='-', linewidth=2, label=f'Max: {max_val:.4f}')
    plt.axvline(np.percentile(df['entropy'], 99), color='orange', linestyle=':', linewidth=2, label='99th Percentile')

    plt.title(f"Распределение предсказаний Энтропии (N={N_SAMPLES})\nWeighted Generator", fontsize=14)
    plt.xlabel("Entropy Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.show()


if __name__ == "__main__":
    main()