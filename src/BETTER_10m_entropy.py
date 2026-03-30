import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# --- НАСТРОЙКИ ---
MODEL_FILENAME = "production_model_v26.pkl"
OUTPUT_CSV = "filtered_sweet_zone.csv"
PLOT_FILENAME = "entropy_sweet_zone.png"

N_SAMPLES = 10_000_000  # 1 миллион точек
BATCH_SIZE = 100_000  # Пачка
THRESHOLD = 0.8

# --- СТРАТЕГИЯ ГЕНЕРАЦИИ ---
# 1. "Магические числа" для Feature 3 (из вашего скриншота)
# Это степени двойки: 2^8 ... 2^25
MAGIC_NUMBERS_F3 = [
    256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152, 4194304,
    8388608, 16777216, 33554432
]

# 2. Диапазон для Feature 4 (сужен до 50 млн, исходя из данных)
RANGE_F4 = (0, 50_000_000)

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

    def fit(self, X_raw, y):
        X_ready = self.preprocessor.transform(X_raw)
        self.model.fit(X_ready, y)
        return self

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        return self.model.predict(X_ready)


# --- ОСНОВНОЙ КОД ---
def main():
    print("=== ГЕНЕРАЦИЯ ПО ШАБЛОНУ 'SWEET ZONE' ===")

    if not os.path.exists(MODEL_FILENAME):
        print(f"❌ Файл {MODEL_FILENAME} не найден!")
        return

    model = joblib.load(MODEL_FILENAME)
    print("✅ Модель загружена.")

    # --- ГЕНЕРАЦИЯ ДАННЫХ ---
    print(f"🎲 Генерация {N_SAMPLES} точек...")
    print("   -> F1 = 1 (Fixed)")
    print("   -> F2 = 0 (Fixed)")
    print(f"   -> F3 = Случайный выбор из {len(MAGIC_NUMBERS_F3)} 'магических' чисел (степени 2)")
    print(f"   -> F4 = Uniform {RANGE_F4} (суженный диапазон)")

    data = {}
    # F1 и F2 фиксированы
    data['feature_1'] = np.full(N_SAMPLES, 1)
    data['feature_2'] = np.full(N_SAMPLES, 0)

    # F3 берем из списка успешных значений
    data['feature_3'] = np.random.choice(MAGIC_NUMBERS_F3, N_SAMPLES)

    # F4 генерируем в диапазоне, где встречается высокая энтропия
    data['feature_4'] = np.random.uniform(RANGE_F4[0], RANGE_F4[1], N_SAMPLES)

    df = pd.DataFrame(data)
    df = df.round(0)

    # Важен порядок колонок!
    df = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']]

    print("✅ Данные готовы. Считаем...")

    # --- ПРОГНОЗ ---
    predictions = []
    for i in tqdm(range(0, N_SAMPLES, BATCH_SIZE), desc="Predicting"):
        batch = df.iloc[i: i + BATCH_SIZE]
        preds_batch = model.predict(batch)
        predictions.extend(preds_batch)

    df['entropy'] = np.array(predictions)

    # --- ГРАФИК ---
    print("📊 Строим график...")
    plt.figure(figsize=(10, 8))

    sns.histplot(y=df['entropy'], bins=100, kde=False, color='red')
    plt.axhline(y=THRESHOLD, color='blue', linestyle='--', linewidth=2, label=f'Target > {THRESHOLD}')

    plt.title(f"Entropy Distribution (Sweet Zone Pattern)\nF3=PowersOf2, F4<50M")
    plt.ylabel("Entropy (Target)")
    plt.xlabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(PLOT_FILENAME)
    print(f"✅ График сохранен: {PLOT_FILENAME}")

    # --- ФИЛЬТРАЦИЯ ---
    print(f"✂️ Отбор значений > {THRESHOLD}...")
    filtered_df = df[df['entropy'] > THRESHOLD]

    rows_saved = len(filtered_df)
    print(f"   Сгенерировано: {N_SAMPLES}")
    print(f"   Прошло фильтр: {rows_saved} ({rows_saved / N_SAMPLES:.2%})")

    if rows_saved > 0:
        # Сортируем по убыванию энтропии (лучшие сверху)
        filtered_df = filtered_df.sort_values(by='entropy', ascending=False)

        print(f"💾 Сохранение в {OUTPUT_CSV}...")
        filtered_df.to_csv(OUTPUT_CSV, sep=';', index=False, decimal=',')

        print("\n🔝 ТОП-5 найденных значений:")
        print(filtered_df.head(5).to_string(index=False))
    else:
        print("⚠️ Ничего не найдено выше порога. Попробуйте чуть опустить THRESHOLD.")


if __name__ == "__main__":
    main()