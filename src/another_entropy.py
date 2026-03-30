import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm

# --- НАСТРОЙКИ ---
MODEL_FILENAME = "production_model_v26.pkl"
OUTPUT_CSV = "filtered_fixed_f1_f2.csv"
PLOT_FILENAME = "entropy_dist_fixed.png"

N_SAMPLES = 1_000_000  # 1 миллион точек
BATCH_SIZE = 100_000  # Пачка
THRESHOLD = 0.9  # Порог

# Диапазоны для генерации оставшихся фичей (3 и 4)
RANGES_DYNAMIC = {
    'feature_3': (1, 2147483648),
    'feature_4': (0, 4294522284)
}

# --- КЛАССЫ (ОБЯЗАТЕЛЬНО ДЛЯ ЗАГРУЗКИ) ---
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
    print("=== ГЕНЕРАЦИЯ (F1=1, F2=0 FIXED) ===")

    # 1. Загрузка
    if not os.path.exists(MODEL_FILENAME):
        print(f"❌ Файл {MODEL_FILENAME} не найден!")
        return

    model = joblib.load(MODEL_FILENAME)
    print("✅ Модель загружена.")

    # 2. Генерация данных
    print(f"🎲 Генерация {N_SAMPLES} точек...")
    data = {}

    # --- ФИКСИРУЕМ ПАРАМЕТРЫ ---
    print("   -> Feature 1 зафиксирован = 1")
    data['feature_1'] = np.full(N_SAMPLES, 1)

    print("   -> Feature 2 зафиксирован = 0")
    data['feature_2'] = np.full(N_SAMPLES, 0)

    # --- ГЕНЕРИРУЕМ ОСТАЛЬНЫЕ (Normal Dist) ---
    print("   -> Feature 3 и 4: Нормальное распределение")
    for col, (min_val, max_val) in RANGES_DYNAMIC.items():
        # Центр диапазона
        mu = (min_val + max_val) / 2
        # Разброс (sigma = 1/6 ширины, чтобы покрыть почти весь диапазон)
        sigma = (max_val - min_val) / 6

        values = np.random.normal(mu, sigma, N_SAMPLES)
        data[col] = np.clip(values, min_val, max_val)

    df = pd.DataFrame(data)
    df = df.round(0)  # Округляем до целых

    # Упорядочиваем колонки (на всякий случай, чтобы порядок был верным)
    df = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']]

    print("✅ Данные готовы. Считаем прогноз...")

    # 3. Прогноз
    predictions = []
    for i in tqdm(range(0, N_SAMPLES, BATCH_SIZE), desc="Calculating"):
        batch = df.iloc[i: i + BATCH_SIZE]
        preds_batch = model.predict(batch)
        predictions.extend(preds_batch)

    df['entropy'] = np.array(predictions)

    # 4. Визуализация (ПОВЕРНУТАЯ)
    print("📊 Строим график...")
    plt.figure(figsize=(10, 8))

    # Горизонтальная гистограмма (y=entropy)
    sns.histplot(y=df['entropy'], bins=100, kde=False, color='crimson')

    plt.axhline(y=THRESHOLD, color='blue', linestyle='--', linewidth=2, label=f'Threshold > {THRESHOLD}')

    plt.title(f"Entropy Distribution (Fixed F1=1, F2=0)\nDynamic F3, F4 (Normal Dist)")
    plt.ylabel("Entropy (Target)")
    plt.xlabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(PLOT_FILENAME)
    print(f"✅ График сохранен: {PLOT_FILENAME}")

    # 5. Фильтрация (> 0.9)
    print(f"✂️ Фильтрация: ищем entropy > {THRESHOLD}")
    filtered_df = df[df['entropy'] > THRESHOLD]

    rows_saved = len(filtered_df)
    print(f"   Сгенерировано: {N_SAMPLES}")
    print(f"   Прошло фильтр: {rows_saved} ({rows_saved / N_SAMPLES:.2%})")

    if rows_saved > 0:
        print(f"💾 Сохранение в {OUTPUT_CSV}...")
        filtered_df.to_csv(OUTPUT_CSV, sep=';', index=False, decimal=',')
        print("🎉 Готово! Результаты сохранены.")
    else:
        print("⚠️ Нет значений > 0.9. Возможно, F3 и F4 тоже нужны ближе к краям, а не в центре диапазона?")


if __name__ == "__main__":
    main()