import pandas as pd
import numpy as np
import joblib
import os
from scipy.optimize import differential_evolution
import warnings

warnings.filterwarnings('ignore')

# --- НАСТРОЙКИ ---
# ВАЖНО: Укажите файл, который создался после калибровки (с байасом 0.1)
# Скорее всего это 'final_model_high_end.joblib' или 'final_model_additive.joblib'
MODEL_FILENAME = "final_model_unbiased.joblib"  # <-- ПРОВЕРЬТЕ ИМЯ ФАЙЛА!

# Диапазоны поиска
VALID_F1 = [1, 2, 4]
VALID_F2 = [0, 3]  # Алгоритм полюбил 3, проверим
VALID_F3 = [
    16384, 32768, 65536, 131072, 262144, 524288,
    1048576, 2097152, 4194304, 8388608, 16777216,
    33554432, 268435456, 536870912, 1073741824
]
F4_MIN = 0
F4_MAX = 300_000_000

# --- КЛАССЫ (НУЖНЫ ДЛЯ ЗАГРУЗКИ) ---
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Базовые классы
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


# Класс Аддитивной модели (если использовали последний скрипт)
class AdditiveCalibratedModel:
    def __init__(self, base_model, bias_value):
        self.base_model = base_model
        self.bias_value = float(bias_value)

    def predict(self, X_raw):
        return self.base_model.predict(X_raw) + self.bias_value


# Класс Регрессионной модели (если использовали предпоследний скрипт)
class CalibratedProductionModel:
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict(self, X_raw):
        raw_pred = self.base_model.predict(X_raw)
        return self.calibrator.predict(raw_pred.reshape(-1, 1))


# --- ПОИСК ---
def main():
    print("=== ПОИСК МАКСИМУМА НА КАЛИБРОВАННОЙ МОДЕЛИ ===")

    if not os.path.exists(MODEL_FILENAME):
        print(f"❌ Файл {MODEL_FILENAME} не найден.")
        return

    model = joblib.load(MODEL_FILENAME)
    print("✅ Модель (с поправкой +0.1) загружена.")

    def objective(x):
        # x[0]=F1_idx, x[1]=F2_idx, x[2]=F3_idx, x[3]=F4_val
        f1 = VALID_F1[int(round(x[0]))]
        f2 = VALID_F2[int(round(x[1]))]
        f3 = VALID_F3[int(round(x[2]))]
        f4 = x[3]

        df = pd.DataFrame({'feature_1': [f1], 'feature_2': [f2], 'feature_3': [f3], 'feature_4': [f4]})
        pred = model.predict(df)[0]
        return -pred  # Минимизируем отрицательное

    bounds = [
        (0, len(VALID_F1) - 1),
        (0, len(VALID_F2) - 1),
        (0, len(VALID_F3) - 1),
        (F4_MIN, F4_MAX)
    ]

    print("⏳ Запуск Differential Evolution...")
    result = differential_evolution(
        objective, bounds, strategy='best1bin', maxiter=30, popsize=10, tol=0.01, disp=True
    )

    print("\n" + "=" * 50)
    print("🚀 ФИНАЛЬНЫЕ ПАРАМЕТРЫ (ДЛЯ БОЕВОГО ЗАПУСКА):")
    print("=" * 50)

    best_x = result.x
    real_pred = -result.fun

    print(f"Feature 1: {VALID_F1[int(round(best_x[0]))]}")
    print(f"Feature 2: {VALID_F2[int(round(best_x[1]))]} (Обратите внимание!)")
    print(f"Feature 3: {VALID_F3[int(round(best_x[2]))]}")
    print(f"Feature 4: {int(best_x[3]):_}")
    print("-" * 30)
    print(f"🔮 ОЖИДАЕМАЯ ЭНТРОПИЯ: {real_pred:.6f}")

    if real_pred > 1.0:
        print("⚠️ Прогноз > 1.0. Это теоретический максимум, модель очень уверена в успехе.")


if __name__ == "__main__":
    main()