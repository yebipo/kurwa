import pandas as pd
import numpy as np
import joblib
import os
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')

# --- НАСТРОЙКИ ---
MODEL_FILENAME = "final_model.joblib"

# Допустимые значения (Discrete)
VALID_F1 = [1, 2, 4]
VALID_F2 = [0, 3]
VALID_F3 = [
    16384, 32768, 65536, 131072, 262144, 524288,
    1048576, 2097152, 4194304, 8388608, 16777216,
    33554432, 268435456, 536870912, 1073741824
]

# Диапазон для F4 (Continuous)
F4_MIN = 0
F4_MAX = 300_000_000  # Ищем в пределах 300 млн

# --- КЛАССЫ ---
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


# --- ЗАГРУЗКА МОДЕЛИ ---
if not os.path.exists(MODEL_FILENAME):
    print("❌ Модель не найдена!")
    exit()

model = joblib.load(MODEL_FILENAME)
print("✅ Модель загружена. Запуск Генетического Алгоритма...")


# --- ФУНКЦИЯ ЦЕЛИ (Оптимизатор пытается сделать её минимальной) ---
# Мы возвращаем "минус энтропию", чтобы найти максимум.
def objective_function(x):
    # x[0] - индекс для F1
    # x[1] - индекс для F2
    # x[2] - индекс для F3
    # x[3] - значение F4

    # Округляем индексы до целых, чтобы брать из списков
    idx_f1 = int(round(x[0]))
    idx_f2 = int(round(x[1]))
    idx_f3 = int(round(x[2]))

    # Защита от выхода за границы списка
    idx_f1 = np.clip(idx_f1, 0, len(VALID_F1) - 1)
    idx_f2 = np.clip(idx_f2, 0, len(VALID_F2) - 1)
    idx_f3 = np.clip(idx_f3, 0, len(VALID_F3) - 1)

    val_f1 = VALID_F1[idx_f1]
    val_f2 = VALID_F2[idx_f2]
    val_f3 = VALID_F3[idx_f3]
    val_f4 = x[3]

    # Собираем DataFrame (1 строка)
    df = pd.DataFrame({
        'feature_1': [val_f1],
        'feature_2': [val_f2],
        'feature_3': [val_f3],
        'feature_4': [val_f4]
    })

    pred = model.predict(df)[0]
    return -pred  # Возвращаем с минусом (ищем минимум минуса = максимум плюса)


# --- ЗАПУСК ОПТИМИЗАЦИИ ---
# Границы поиска:
# F1: индекс от 0 до len-1
# F2: индекс от 0 до len-1
# F3: индекс от 0 до len-1
# F4: значение от MIN до MAX
bounds = [
    (0, len(VALID_F1) - 1),
    (0, len(VALID_F2) - 1),
    (0, len(VALID_F3) - 1),
    (F4_MIN, F4_MAX)
]

print("⏳ Идет поиск (это займет 1-2 минуты)...")

# Differential Evolution - мощный метод глобальной оптимизации
result = differential_evolution(
    objective_function,
    bounds,
    strategy='best1bin',
    maxiter=50,  # Количество поколений
    popsize=15,  # Размер популяции
    tol=0.01,
    disp=True  # Показывает прогресс
)

# --- РЕЗУЛЬТАТ ---
print("\n" + "=" * 50)
print("💎 НАЙДЕН АБСОЛЮТНЫЙ МАКСИМУМ ЭТОЙ МОДЕЛИ")
print("=" * 50)

best_x = result.x
max_entropy = -result.fun  # Убираем минус обратно

best_f1 = VALID_F1[int(round(best_x[0]))]
best_f2 = VALID_F2[int(round(best_x[1]))]
best_f3 = VALID_F3[int(round(best_x[2]))]
best_f4 = best_x[3]

print(f"🔥 MAX Entropy: {max_entropy:.6f}")
print("-" * 30)
print(f"👉 Feature 1: {best_f1}")
print(f"👉 Feature 2: {best_f2}")
print(f"👉 Feature 3: {best_f3}")
print(f"👉 Feature 4: {best_f4:,.2f}")
print("=" * 50)

if max_entropy < 0.9:
    print("⚠️ ВЕРДИКТ: Модель не знает значений > 0.9.")
    print("   Она выучила 'среднее по больнице', но сгладила высокие пики.")
    print("   Чтобы получить 0.95, нужно переобучить модель с другими настройками")
    print("   (например, убрать L2-регуляризацию или увеличить глубину деревьев).")