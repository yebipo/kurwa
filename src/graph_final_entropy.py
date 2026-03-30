import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm  # pip install tqdm

# --- НАСТРОЙКИ ---
MODEL_FILENAME = "production_model_v26.pkl"
OUTPUT_CSV = "filtered_entropy_HIGH.csv"
PLOT_FILENAME = "entropy_distribution_huge.png"

N_SAMPLES = 10_000_000  # 10 млн точек
BATCH_SIZE = 100_000  # Размер пачки для прогноза
THRESHOLD = 0.9  # Порог

# Ваши диапазоны
RANGES = {
    'feature_1': (1, 2147483648),
    'feature_2': (0, 4294907120),
    'feature_3': (1, 2147483648),
    'feature_4': (0, 4294522284)
}

# --- КЛАССЫ (ОБЯЗАТЕЛЬНО ДЛЯ JOBLIB) ---
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
    print("=== ПОИСК ВЫСОКОЙ ЭНТРОПИИ (> 0.9) ===")

    # 1. Загрузка
    if not os.path.exists(MODEL_FILENAME):
        print(f"❌ Файл {MODEL_FILENAME} не найден!")
        return

    print("⏳ Загрузка модели...")
    model = joblib.load(MODEL_FILENAME)
    print("✅ Модель в памяти.")

    # 2. Генерация данных
    print(f"🎲 Генерация {N_SAMPLES} точек...")
    data = {}

    for col, (min_val, max_val) in RANGES.items():
        # Равномерное распределение в огромных диапазонах
        data[col] = np.random.uniform(min_val, max_val, N_SAMPLES)

    df = pd.DataFrame(data)
    df = df.round(0)  # Округляем до целых

    print("✅ Генерация завершена. Старт предсказания...")

    # 3. Прогноз (Batch processing)
    predictions = []
    for i in tqdm(range(0, N_SAMPLES, BATCH_SIZE), desc="Calculation"):
        batch = df.iloc[i: i + BATCH_SIZE]
        preds_batch = model.predict(batch)
        predictions.extend(preds_batch)

    df['entropy'] = np.array(predictions)

    # 4. Визуализация
    print("📊 Строим график...")
    plt.figure(figsize=(10, 6))

    # Гистограмма
    sns.histplot(df['entropy'], bins=100, kde=False, color='orange')

    # Линия отсечения
    plt.axvline(x=THRESHOLD, color='blue', linestyle='--', linewidth=3, label=f'Cutoff > {THRESHOLD}')

    plt.title(f"Распределение энтропии (N={N_SAMPLES})")
    plt.xlabel("Entropy Value")
    plt.ylabel("Количество точек")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(PLOT_FILENAME)
    print(f"✅ График сохранен: {PLOT_FILENAME}")
    # plt.show() # Можно раскомментировать, если хотите видеть окно с графиком

    # 5. Фильтрация (>= 0.9)
    print(f"Ищем entropy > {THRESHOLD} ...")

    filtered_df = df[df['entropy'] >= THRESHOLD]

    rows_saved = len(filtered_df)
    print(f"   Всего сгенерировано: {N_SAMPLES}")
    print(f"   Прошло фильтр (> {THRESHOLD}): {rows_saved} ({rows_saved / N_SAMPLES:.2%})")

    if rows_saved > 0:
        print(f"Сохранение в {OUTPUT_CSV}...")
        filtered_df.to_csv(OUTPUT_CSV, sep=';', index=False, decimal=',')
        print("Готово!")
    else:
        print(
            "⚠️ Внимание: Ни одна точка не превысила порог 0.9! Возможно, таких значений просто нет в этом диапазоне.")


if __name__ == "__main__":
    main()