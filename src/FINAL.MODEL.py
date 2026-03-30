import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

# --- НАСТРОЙКИ ---
INPUT_FILE = 'million.csv'
# Имя файла должно совпадать с тем, что ты создал в предыдущем шаге
MODEL_FILENAME = "final_model.joblib"
RANDOM_STATE = 42

# Диапазон анализа
RANGE_MIN = 0.80
RANGE_MAX = 1.00


# --- 1. ВОССТАНОВЛЕНИЕ КЛАССОВ (COPY-PASTE) ---
# Это обязательно, чтобы joblib понял структуру объекта

class HighEndQuantileMapper:
    def __init__(self, n_bins=1000):
        self.n_bins = n_bins
        self.x_ref = None;
        self.y_ref = None

    def fit(self, y_pred, y_real): pass  # Логика не нужна для загрузки

    def predict(self, y_pred): return np.interp(y_pred, self.x_ref, self.y_ref)


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
        self.estimators = [('lgbm', lgb.LGBMRegressor(**lgbm_p)), ('xgb', xgb.XGBRegressor(**xgb_p)),
                           ('catboost', cb.CatBoostRegressor(**cat_p))]
        self.model = StackingRegressor(estimators=self.estimators, final_estimator=LinearRegression(), cv=3, n_jobs=1)

    def fit(self, X_raw, y): pass

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        return self.model.predict(X_ready)


class FinalQuantileWrapper:
    def __init__(self, base_model, mapper, threshold=0.85):
        self.base_model = base_model
        self.mapper = mapper
        self.threshold = threshold

    def predict(self, X):
        preds = self.base_model.predict(X)
        preds = np.clip(preds, 0, 1)
        mask_high = preds > self.threshold
        if mask_high.sum() > 0:
            high_preds = preds[mask_high]
            corrected = self.mapper.predict(high_preds)
            preds[mask_high] = corrected
        return preds


# Хак для Joblib, если файл был сохранен из __main__
import types

module_fake = types.ModuleType('__main__')
module_fake.HighEndQuantileMapper = HighEndQuantileMapper
module_fake.GlobalPreprocessor = GlobalPreprocessor
module_fake.ProductionModel = ProductionModel
module_fake.FinalQuantileWrapper = FinalQuantileWrapper


# Если вдруг сохранял не из __main__, а из именованного модуля, раскомментируй и поправь:
# sys.modules['имя_твоего_скрипта_обучения'] = module_fake

# --- 2. ФУНКЦИИ АНАЛИЗА ---

def main():
    print(f"=== CHECKING NEW MODEL: {MODEL_FILENAME} ===")
    print(f"=== ANALYSIS RANGE: {RANGE_MIN} - {RANGE_MAX} ===\n")

    # 1. Загрузка данных
    if not os.path.exists(INPUT_FILE) or not os.path.exists(MODEL_FILENAME):
        print("❌ Файлы не найдены.")
        return

    print("📂 Чтение данных...")
    df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)

    X = df.drop(columns=['target'])
    y = df['target'].values

    # Делаем сплит, чтобы проверить на отложенной выборке (как по науке)
    # Если хочешь проверить на том же, на чем учил (bias fit), убери сплит
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 2. Фильтрация зоны интереса (0.8 - 1.0)
    mask_range = (y_test >= RANGE_MIN) & (y_test <= RANGE_MAX)
    X_target = X_test[mask_range]
    y_target = y_test[mask_range]

    print(f"🎯 Точек в диапазоне: {len(y_target)}")

    # 3. Загрузка и Предикт
    print("🔄 Загрузка модели...")
    try:
        final_model = joblib.load(MODEL_FILENAME)
        print("   ✅ Модель загружена.")
    except Exception as e:
        print(f"❌ Ошибка загрузки (проверь классы): {e}")
        return

    print("⚙️ Расчет предсказаний...")

    # А) Базовый прогноз (БЕЗ маппера) - достаем внутреннюю модель
    print("   -> Raw Prediction (Base Model)...")
    preds_raw = final_model.base_model.predict(X_target)
    preds_raw = np.clip(preds_raw, 0, 1)

    # Б) Финальный прогноз (С маппером) - вызываем обертку
    print("   -> Final Prediction (Quantile Corrected)...")
    preds_final = final_model.predict(X_target)

    # 4. Метрики
    def get_metrics(y_true, y_pred, name):
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        bias = np.mean(y_pred - y_true)
        max_val = np.max(y_pred)
        print(f"{name:<25} | RMSE: {rmse:.5f} | Bias: {bias:.5f} | Max: {max_val:.5f}")

    print("\n📊 РЕЗУЛЬТАТЫ:")
    print("-" * 75)
    get_metrics(y_target, preds_raw, "1. Base Stacking")
    get_metrics(y_target, preds_final, "2. Quantile Wrapper")
    print("-" * 75)

    # 5. Графики
    plt.figure(figsize=(16, 7))

    # --- График 1: Гистограммы Ошибок ---
    plt.subplot(1, 2, 1)
    sns.kdeplot(preds_raw - y_target, label='Base Model', color='orange', fill=True, alpha=0.3)
    sns.kdeplot(preds_final - y_target, label='Quantile Fixed', color='crimson', fill=True, alpha=0.3)
    plt.axvline(0, color='k', linestyle='--', linewidth=1)
    plt.title(f"Bias Distribution ({RANGE_MIN}-{RANGE_MAX})")
    plt.xlabel("Error (Pred - Real)")
    plt.legend()

    # --- График 2: Scatter ---
    plt.subplot(1, 2, 2)
    # Рисуем Base
    plt.scatter(y_target, preds_raw, s=15, alpha=0.2, c='orange', label='Base Model')
    # Рисуем Final
    plt.scatter(y_target, preds_final, s=15, alpha=0.2, c='crimson', label='Quantile Fixed')

    # Идеал
    plt.plot([RANGE_MIN, 1], [RANGE_MIN, 1], 'k--', lw=2, label='Ideal')

    plt.xlim(RANGE_MIN, 1.02)
    plt.ylim(RANGE_MIN, 1.02)
    plt.title("Real vs Predicted (Zoomed)")
    plt.xlabel("Real Target")
    plt.ylabel("Prediction")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()