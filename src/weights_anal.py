import pandas as pd
import numpy as np
import joblib
import os
from tqdm import tqdm

# --- НАСТРОЙКИ ---
MODEL_FILENAME = "production_model_v26.pkl"
OUTPUT_CSV = "final_weighted_results.csv"

N_SAMPLES = 2_000_000  # Генерируем 2 миллиона, так как веса точные
BATCH_SIZE = 100_000
THRESHOLD = 0.9  # Поднимаем планку, ищем самую элиту

# --- ВЕСА (Из вашего лога) ---

# Feature 1 (Col_2)
F1_VALS = [1, 2, 4]
F1_PROBS = [0.9481, 0.0447, 0.0072]

# Feature 2 (Col_3)
F2_VALS = [0, 3]
F2_PROBS = [0.9603, 0.0397]

# Feature 3 (Col_4) - Степени двойки
F3_VALS = [16777216, 536870912, 2097152, 262144, 4194304, 32768, 1048576, 512, 4096, 2048, 524288, 131072, 256, 8192,
           268435456, 128, 1073741824, 16384]
F3_PROBS = [0.139, 0.1334, 0.1096, 0.0873, 0.0852, 0.0722, 0.0697, 0.057, 0.0506, 0.047, 0.0466, 0.0433, 0.0243, 0.0229,
            0.0076, 0.0026, 0.0013, 0.0004]

# Feature 4 (Col_5) - Непрерывное
# 89% данных лежит до 291 млн. Медиана 750к.
# Генерируем Log-Uniform, чтобы захватить и маленькие (700к) и большие (300м) числа
F4_MIN = 1000  # Ставим разумный минимум
F4_MAX = 300_000_000  # Ставим разумный максимум (зона 89%)


# --- ФУНКЦИЯ НОРМАЛИЗАЦИИ ВЕРОЯТНОСТЕЙ ---
def normalize_probs(probs):
    """Гарантирует, что сумма равна ровно 1.0, иначе numpy ругается"""
    p = np.array(probs)
    return p / p.sum()


# --- КЛАССЫ МОДЕЛИ ---
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
    print(f"=== ГЕНЕРАЦИЯ ПО ВЕСАМ (TARGET > {THRESHOLD}) ===")

    # 1. Загрузка
    if not os.path.exists(MODEL_FILENAME):
        print("❌ Нет файла модели.")
        return
    model = joblib.load(MODEL_FILENAME)
    print("✅ Модель загружена.")

    # 2. Генерация данных
    print(f"🎲 Создание {N_SAMPLES} образцов на основе статистики...")

    # Нормализуем вероятности (чтобы сумма была 1.0)
    p1 = normalize_probs(F1_PROBS)
    p2 = normalize_probs(F2_PROBS)
    p3 = normalize_probs(F3_PROBS)

    data = {
        'feature_1': np.random.choice(F1_VALS, size=N_SAMPLES, p=p1),
        'feature_2': np.random.choice(F2_VALS, size=N_SAMPLES, p=p2),
        'feature_3': np.random.choice(F3_VALS, size=N_SAMPLES, p=p3),
        # Используем логарифмическое распределение для Feature 4
        # Это дает больше чисел около медианы (750к), но достает и до 300М
        'feature_4': np.exp(np.random.uniform(np.log(F4_MIN), np.log(F4_MAX), N_SAMPLES))
    }

    df = pd.DataFrame(data).round(0)
    # Порядок колонок важен для модели!
    df = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']]

    print("✅ Данные сгенерированы. Старт оценки...")

    # 3. Предсказание
    predictions = []
    for i in tqdm(range(0, N_SAMPLES, BATCH_SIZE), desc="Predicting"):
        batch = df.iloc[i: i + BATCH_SIZE]
        preds = model.predict(batch)
        predictions.extend(preds)

    df['entropy'] = np.array(predictions)

    # 4. Фильтрация и сохранение
    print(f"✂️ Фильтрация лучших результатов (> {THRESHOLD})...")
    best_df = df[df['entropy'] > THRESHOLD].copy()

    # Сортировка (лучшие сверху)
    best_df = best_df.sort_values(by='entropy', ascending=False)

    count = len(best_df)
    print(f"🏆 Найдено {count} строк с высокой энтропией!")

    if count > 0:
        print(f"💾 Сохранение в {OUTPUT_CSV}...")
        best_df.to_csv(OUTPUT_CSV, sep=';', decimal=',', index=False)

        print("\n🔝 ТОП-10 НАЙДЕННЫХ ПАРАМЕТРОВ:")
        print(best_df.head(10).to_string(index=False))
    else:
        print("🤷 Ничего не найдено выше порога. Попробуйте снизить THRESHOLD или расширить диапазоны.")


if __name__ == "__main__":
    main()