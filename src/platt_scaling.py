import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from scipy.optimize import minimize

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Импорты для того, чтобы joblib знал, как загружать модели
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

# --- НАСТРОЙКИ ---
INPUT_FILE = 'unique_rows.csv'
BASE_MODEL_FILE = "final_model_unbiased.joblib"
OUTPUT_MODEL_FILE = "final_model_smooth.joblib"

CSV_SEP = ';'
CSV_DECIMAL = ','


# --- КЛАССЫ (ОБЯЗАТЕЛЬНО ДЛЯ ЗАГРУЗКИ МОДЕЛИ) ---
class GlobalPreprocessor:
    def __init__(self, scaler, kmeans):
        self.scaler = scaler
        self.kmeans = kmeans

    def transform(self, X_raw):
        # Превращаем в float и обрабатываем логарифмом
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

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        return self.model.predict(X_ready)


# 🆕 НОВЫЙ ПЛАВНЫЙ КАЛИБРАТОР (PLATT SCALING)
class PlattCalibrator:
    def __init__(self):
        # Коэффициенты a и b в формуле сигмоиды: y = 1 / (1 + exp(-(a*x + b)))
        self.a_ = None
        self.b_ = None

    def fit(self, X, y_true):
        # Убедимся, что y_true [0, 1] и немного сгладим края для стабильности
        y_safe = np.clip(y_true, 1e-7, 1 - 1e-7)
        # Рассчитываем логарифм шансов (logit transformation) для таргета
        y_trans = np.log(y_safe / (1 - y_safe))

        # Обучаем простую линейную регрессию, чтобы найти начальные ax + b
        # Но для минимизации лог-лосса (Log-Loss) используем оптимизатор
        X_flat = X.flatten()

        # Функция лог-лосса для минимизации
        def log_loss(params, X, y_true):
            a, b = params
            linear = a * X + b
            prob = 1.0 / (1.0 + np.exp(-linear))
            return -np.mean(y_true * np.log(prob + 1e-10) + (1.0 - y_true) * np.log(1.0 - prob + 1e-10))

        # Начнем с коэффициентов от линейной регрессии
        from sklearn.linear_model import LinearRegression as SkLinearRegression
        lr = SkLinearRegression().fit(X_flat.reshape(-1, 1), y_trans)
        initial_params = np.array([lr.coef_[0], lr.intercept_])

        # Оптимизация
        result = minimize(log_loss, initial_params, args=(X_flat, y_safe), method='BFGS')
        self.a_, self.b_ = result.x
        print(f"✅ Коэффициенты сигмоиды найдены: a={self.a_:.4f}, b={self.b_:.4f}")
        return self

    def predict(self, X):
        # 1D массив на вход
        linear = self.a_ * X.flatten() + self.b_
        # Применяем логистическую (сигмоидную) функцию
        smooth_prob = 1.0 / (1.0 + np.exp(-linear))
        # Ограничиваем [0, 1] и возвращаем ту же форму, что и на входе
        return np.clip(smooth_prob, 0, 1).reshape(X.shape)


class CalibratedProductionModel:
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict(self, X_raw):
        # 1. Получаем сырой прогноз
        raw_pred = self.base_model.predict(X_raw)
        # 2. Прогоняем через плавную сигмоиду
        # Новая реализация predict принимает тот же массив, что и на входе
        return self.calibrator.predict(raw_pred)


# --- ФУНКЦИИ ---
def clean_and_load_data(file_path):
    print(f"🧹 Зачистка данных в {file_path}...")
    try:
        # Читаем с твоими разделителями
        df = pd.read_csv(file_path, sep=CSV_SEP, decimal=CSV_DECIMAL, header=None, on_bad_lines='skip', engine='python')

        # Берем колонки 1,2,3,4 (фичи) и 5 (таргет)
        df = df[[1, 2, 3, 4, 5]]

        # Называем их ровно так, как видела модель при обучении
        df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']

        # Чистим от не-числовых значений и NaN
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        initial_len = len(df)
        df.dropna(inplace=True)

        print(f"✅ Готово. Строк для анализа: {len(df)} (отсеяно {initial_len - len(df)})")
        return df
    except Exception as e:
        print(f"❌ Ошибка при чтении файла: {e}")
        return None


def main():
    # 1. Загрузка
    df = clean_and_load_data(INPUT_FILE)
    if df is None: return

    X = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']]
    y = df['target'].values

    # Сплит: 50% на обучение "растягивалки", 50% на честный тест
    X_cal, X_test, y_cal, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    print(f"✂️ Выборка: {len(y_cal)} на калибровку, {len(y_test)} на тест.")

    # 2. Загрузка базовой модели
    if not os.path.exists(BASE_MODEL_FILE):
        print(f"❌ Модель {BASE_MODEL_FILE} не найдена в папке.")
        return

    print(f"🔄 Загрузка {BASE_MODEL_FILE}...")
    loaded_obj = joblib.load(BASE_MODEL_FILE)

    # Достаем саму модель (стекинг) из обертки, если она там есть
    base_model = loaded_obj.base_model if hasattr(loaded_obj, 'base_model') else loaded_obj

    # 3. 🆕 Обучение PlattCalibrator (Плавно!)
    print("📏 Настраиваем плавное сигмоидное растяжение...")
    y_raw_cal = base_model.predict(X_cal)

    # Создаем и обучаем калибратор
    smooth_cal = PlattCalibrator()
    smooth_cal.fit(y_raw_cal, y_cal)

    # 4. Оценка результатов
    print("🧪 Проверка на тестовых данных...")
    y_raw_test = base_model.predict(X_test)
    y_final_test = smooth_cal.predict(y_raw_test)

    rmse_raw = np.sqrt(mean_squared_error(y_test, y_raw_test))
    rmse_final = np.sqrt(mean_squared_error(y_test, y_final_test))

    print("\n" + "=" * 45)
    print(f"📉 RMSE ДО калибровки:    {rmse_raw:.6f}")
    print(f"📈 RMSE ПОСЛЕ (Platt):    {rmse_final:.6f}")
    profit = (rmse_raw - rmse_final) / rmse_raw * 100
    print(f"🚀 Профит точности:       {profit:.2f}%")
    print("=" * 45)

    # 5. Сохранение
    print(f"💾 Сохранение новой модели в {OUTPUT_MODEL_FILE}...")
    final_model_obj = CalibratedProductionModel(base_model, smooth_cal)
    joblib.dump(final_model_obj, OUTPUT_MODEL_FILE)

    # 6. Графики
    plt.figure(figsize=(16, 7))

    # Левый график: Scatter (наглядное растяжение)
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_raw_test, s=2, alpha=0.1, color='orange', label='Raw (Squeezed)')
    plt.scatter(y_test, y_final_test, s=2, alpha=0.1, color='blue', label='Platt Smooth (Stretched)')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Ideal')
    plt.title("Real vs Predicted: Smooth Stretching Effect")
    plt.xlabel("Real Target")
    plt.ylabel("Prediction")
    plt.legend()

    # Правый график: Ошибки (Bias)
    plt.subplot(1, 2, 2)
    sns.kdeplot(y_raw_test - y_test, color='orange', fill=True, label='Error: Raw')
    sns.kdeplot(y_final_test - y_test, color='blue', fill=True, label='Error: Platt Smooth')
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Error Distribution (Bias Correction)")
    plt.xlabel("Prediction Error")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()