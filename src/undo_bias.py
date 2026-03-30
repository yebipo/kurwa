import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# --- НАСТРОЙКИ ---
INPUT_FILE = 'best_entropy.csv'  # Файл с реальными данными
BASE_MODEL_FILE = "final_model.joblib"  # Ваша текущая (занижающая) модель
FINAL_MODEL_FILE = "final_model_unbiased.joblib"  # Куда сохранить исправленную

CSV_SEP = ';'
CSV_DECIMAL = ','

# --- КЛАССЫ (ОБЯЗАТЕЛЬНО ДЛЯ ЗАГРУЗКИ СТАРОЙ МОДЕЛИ) ---
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import StackingRegressor


# ... (импорты внутри классов уже есть в памяти, но для скрипта нужны)

# 1. Определяем классы для старой модели
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


# 2. 🆕 НОВЫЙ КЛАСС-ОБЕРТКА (Супер-модель)
class CalibratedProductionModel:
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict(self, X_raw):
        # 1. Получаем сырой прогноз (заниженный)
        raw_pred = self.base_model.predict(X_raw)
        # 2. Прогоняем через регрессию (исправляем)
        # reshape(-1, 1) нужен, так как sklearn ждет 2D массив
        final_pred = self.calibrator.predict(raw_pred.reshape(-1, 1))
        return final_pred


# --- MAIN ---
def main():
    print("=== ОБУЧЕНИЕ КАЛИБРАТОРА (LINEAR REGRESSION) ===")

    # 1. Загрузка данных
    print(f"📂 Читаем {INPUT_FILE}...")
    try:
        df = pd.read_csv(INPUT_FILE, sep=CSV_SEP, decimal=CSV_DECIMAL, header=None, skiprows=1)
        df.rename(columns={1: 'feature_1', 2: 'feature_2', 3: 'feature_3', 4: 'feature_4', 5: 'target'}, inplace=True)
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df.dropna(subset=['target'], inplace=True)

        # Используем ВСЕ данные для обучения регрессии, чтобы она была устойчивой
        X = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']]
        y_true = df['target']
        print(f"   Загружено строк: {len(df)}")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return

    # 2. Загрузка старой модели
    print(f"⏳ Загрузка {BASE_MODEL_FILE}...")
    base_model = joblib.load(BASE_MODEL_FILE)

    # 3. Получение "сырых" прогнозов
    print("🧠 Генерация базовых предсказаний...")
    y_raw_pred = base_model.predict(X)

    # 4. Обучение Калибратора
    # Задача: Найти A и B, такие что: Y_true = A * Y_raw + B
    print("📐 Обучение линейной коррекции...")

    calibrator = LinearRegression()
    # Вход: сырые прогнозы (X), Выход: реальность (y)
    calibrator.fit(y_raw_pred.reshape(-1, 1), y_true)

    coef_a = calibrator.coef_[0]
    coef_b = calibrator.intercept_

    print("\n" + "=" * 40)
    print("🎯 ФОРМУЛА ИСПРАВЛЕНИЯ:")
    print(f"   Final = {coef_a:.4f} * Raw_Prediction + {coef_b:.4f}")
    print("=" * 40)

    if coef_a > 1.0:
        print("👉 Коэффициент > 1.0: Модель 'растягивает' пики вверх (как мы и хотели).")

    # 5. Проверка
    y_calibrated = calibrator.predict(y_raw_pred.reshape(-1, 1))

    # Сравниваем ошибку на ТОП-1000 значений (самых высоких)
    # Нам важно, чтобы ошибка уменьшилась именно там
    top_indices = y_true.nlargest(1000).index

    bias_raw = (y_true[top_indices] - y_raw_pred[top_indices]).mean()
    bias_calib = (y_true[top_indices] - y_calibrated[top_indices]).mean()

    print(f"\n📊 Проверка на ТОП-1000 высоких значений:")
    print(f"   Средний Bias ДО:    {bias_raw:+.6f}")
    print(f"   Средний Bias ПОСЛЕ: {bias_calib:+.6f}")

    # 6. Сохранение новой модели
    print(f"\n💾 Создание и сохранение {FINAL_MODEL_FILE}...")
    final_model_obj = CalibratedProductionModel(base_model, calibrator)
    joblib.dump(final_model_obj, FINAL_MODEL_FILE)
    print("✅ Готово! Теперь используйте этот файл для прогнозов.")

    # 7. График
    plt.figure(figsize=(10, 6))
    # Рисуем только сэмпл, чтобы не тормозило
    sample_idx = np.random.choice(len(y_true), min(5000, len(y_true)), replace=False)

    plt.scatter(y_raw_pred[sample_idx], y_true.iloc[sample_idx], alpha=0.3, s=10, label='Raw Predictions', color='gray')
    plt.scatter(y_calibrated[sample_idx], y_true.iloc[sample_idx], alpha=0.3, s=10, label='Calibrated', color='blue')

    # Идеальная линия
    min_val = min(y_true.min(), y_calibrated.min())
    max_val = max(y_true.max(), y_calibrated.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal (y=x)')

    plt.title('Calibration Effect: Raw vs Calibrated')
    plt.xlabel('Predicted Entropy')
    plt.ylabel('Real Entropy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()