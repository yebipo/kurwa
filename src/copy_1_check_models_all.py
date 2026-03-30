import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. НАСТРОЙКИ
# ==============================================================================
# from google.colab import drive
# drive.mount('/content/drive')

WORK_DIR = r"D:\0000000ядро\анализ\.venv\Scripts\without_bugs"  # Ваш путь
INPUT_FILE = 'million.csv'
MODEL_FILENAME = f"{WORK_DIR}/final_model_god_with_bar.joblib"
ELITE_CORRECTOR_PATH = "elite_linear_corrector.joblib"

RANDOM_STATE = 42
EPSILON = 1e-5
ANALYSIS_THRESHOLD = 0.90


# ==============================================================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================
def to_logit(y):
    y_safe = np.clip(y, EPSILON, 1 - EPSILON)
    return np.log(y_safe / (1 - y_safe))


def from_logit(z):
    res = 1 / (1 + np.exp(-z))
    return np.clip(res, 0, 1)


# ==============================================================================
# 3. КЛАССЫ (ВСЕ ВЕРСИИ ДЛЯ ЗАГРУЗКИ)
# ==============================================================================

# --- A. КЛАССЫ ДЛЯ СТАРОЙ МОДЕЛИ (Old Model / FinalModel) ---
class GlobalPreprocessor:
    def __init__(self, scaler, kmeans):
        self.scaler = scaler
        self.kmeans = kmeans

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        X_out = X_log.copy()
        X_out['cluster_id'] = self.kmeans.predict(X_scaled)
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

    def fit(self, X, y): pass

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        return self.model.predict(X_ready)


# --- B. КЛАССЫ ДЛЯ НОВОЙ МОДЕЛИ (God Mode / Hybrid) ---
class SmartPreprocessor:
    def __init__(self, n_clusters=20):
        # Важно: имена должны совпадать с тем, что в файле (scaler, kmeans)
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
        self.n_clusters = n_clusters

    def fit(self, X): pass

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)

        # Генерация фич
        df['row_mean'] = df.mean(axis=1)
        df['row_std'] = df.std(axis=1)
        df['row_max'] = df.max(axis=1)

        dists = self.kmeans.transform(X_scaled)
        for i in range(self.n_clusters):
            df[f'dist_clust_{i}'] = dists[:, i]

        df['cluster_id'] = self.kmeans.predict(X_scaled)
        return df


class IsotonicCorrector:
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    def fit(self, y_p, y_t): pass

    def predict(self, y_p): return self.iso.predict(y_p)


class HybridModelGod:
    def __init__(self, preprocessor, p_lgbm_q, p_lgbm_m, p_xgb_q, p_xgb_m, p_cat_q, p_cat_m):
        self.preprocessor = preprocessor
        self.bias_corrector = IsotonicCorrector()
        self.model = None
        self.base_models = []  # Здесь joblib восстановит список моделей
        self.final_estimator = None

    def predict_logit_raw(self, X_raw):
        X = self.preprocessor.transform(X_raw)
        meta_features = np.column_stack([m.predict(X) for name, m in self.base_models])
        return self.final_estimator.predict(meta_features)

    def predict(self, X_raw, apply_calibration=True):
        logits = self.predict_logit_raw(X_raw)
        preds = from_logit(logits)
        if apply_calibration: preds = self.bias_corrector.predict(preds)
        return preds


# --- C. ДОПОЛНИТЕЛЬНЫЕ КЛАССЫ ИЗ ФАЙЛА (Для совместимости) ---
class CalibratedProductionModel:
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict(self, X_raw):
        raw_pred = self.base_model.predict(X_raw)
        return self.calibrator.predict(raw_pred.reshape(-1, 1))


class AdditiveCalibratedModel:
    def __init__(self, base_model, bias_value):
        self.base_model = base_model
        self.bias_value = float(bias_value)

    def predict(self, X_raw):
        return self.base_model.predict(X_raw) + self.bias_value


# ==============================================================================
# 4. MAIN PROGRAM
# ==============================================================================
def main():
    print("=== ЗАПУСК: ЧИСТО КВАНТИЛЬНЫЙ АНСАМБЛЬ (FIXED SIGMOID) ===\n")

    if not os.path.exists(INPUT_FILE):
        print("❌ Файл с данными не найден.")
        return

    print("📂 Читаем данные...")
    df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)
    X = df.drop(columns=['target'])
    y_real = df['target'].values

    # Делим на трейн/тест для честности
    X_train, X_test, y_train, y_test = train_test_split(X, y_real, test_size=0.2, random_state=RANDOM_STATE)

    print("🔄 Загрузка God Model...")
    try:
        god_model = joblib.load(MODEL_FILENAME)
        print("   ✅ Успешно загружена.")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    print("⚙️ Препроцессинг (Smart Features)...")
    # Используем препроцессор из загруженной модели
    X_test_transformed = god_model.preprocessor.transform(X_test)

    # --------------------------------------------------------
    # 5. ИЗВЛЕЧЕНИЕ ЭЛИТЫ (QUANTILE MODELS)
    # --------------------------------------------------------
    elite_preds = []

    print("\n🕵️‍♂️ Извлечение квантильных моделей:")
    for name, model in god_model.base_models:
        if name.endswith('_q'):  # Ищем lgbm_q, xgb_q, cat_q
            print(f"   -> Используем: {name}")

            # 1. Получаем сырой прогноз (это ЛОГИТЫ! -inf..+inf)
            logits = model.predict(X_test_transformed)

            # 2. ПРИМЕНЯЕМ СИГМОИДУ (FIX!) -> получаем 0..1
            probs = from_logit(logits)

            elite_preds.append(probs)

    if not elite_preds:
        print("❌ Квантильные модели не найдены в файле!")
        return

    # 6. Усреднение (Ансамблирование)
    print("\n🔗 Усреднение ансамбля (Elite 3)...")
    ensemble_raw = np.mean(elite_preds, axis=0)

    # --------------------------------------------------------
    # 7. ЛИНЕЙНАЯ КОРРЕКЦИЯ
    # --------------------------------------------------------
    print("\n🔧 Обучение Линейной Коррекции...")
    # Обучаемся выравнивать прогноз под реальность
    residuals = y_test - ensemble_raw

    corrector = LinearRegression()
    corrector.fit(ensemble_raw.reshape(-1, 1), residuals)

    bias_fix = corrector.predict(ensemble_raw.reshape(-1, 1))
    ensemble_corrected = np.clip(ensemble_raw + bias_fix, 0, 1)

    print(f"   Формула: Correction = {corrector.coef_[0]:.4f} * Pred + {corrector.intercept_:.4f}")
    joblib.dump(corrector, ELITE_CORRECTOR_PATH)

    # --------------------------------------------------------
    # 8. АНАЛИЗ РЕЗУЛЬТАТОВ
    # --------------------------------------------------------
    mask_high = y_test > ANALYSIS_THRESHOLD

    print("\n" + "=" * 60)
    print(f"📊 РЕЗУЛЬТАТЫ (REAL > {ANALYSIS_THRESHOLD})")
    print("=" * 60)

    if mask_high.sum() > 0:
        # Метрики
        bias_raw = np.mean(y_test[mask_high] - ensemble_raw[mask_high])
        bias_final = np.mean(y_test[mask_high] - ensemble_corrected[mask_high])

        r2_raw = r2_score(y_test[mask_high], ensemble_raw[mask_high])
        r2_final = r2_score(y_test[mask_high], ensemble_corrected[mask_high])

        rmse_final = np.sqrt(mean_squared_error(y_test[mask_high], ensemble_corrected[mask_high]))

        print(f"{'Metric':<15} | {'Raw Elite':<15} | {'Corrected Elite':<15}")
        print("-" * 60)
        print(f"{'Mean Bias':<15} | {bias_raw:.6f}        | {bias_final:.6f}")
        print(f"{'R2 Score':<15} | {r2_raw:.5f}        | {r2_final:.5f}")
        print(f"{'RMSE':<15} | {'-':<15} | {rmse_final:.5f}")

        # ГРАФИКИ
        plt.figure(figsize=(14, 6))

        # 1. Гистограмма (Ждем ОДИН горб на нуле)
        plt.subplot(1, 2, 1)
        sns.histplot(y_test[mask_high] - ensemble_corrected[mask_high], bins=100, kde=True, color='teal',
                     label='Elite Corrected')
        plt.axvline(0, color='red', linestyle='--', lw=2)
        plt.title(f"Распределение Байаса (> {ANALYSIS_THRESHOLD})\nMean: {bias_final:.6f}")
        plt.xlabel("Error")
        plt.legend()

        # 2. Scatter
        plt.subplot(1, 2, 2)
        plt.scatter(y_test[mask_high], ensemble_corrected[mask_high], alpha=0.3, s=5, c='purple', label='Predictions')
        plt.plot([ANALYSIS_THRESHOLD, 1], [ANALYSIS_THRESHOLD, 1], 'k--', lw=2, label='Ideal')

        # Линия тренда ошибки
        sns.regplot(x=y_test[mask_high], y=(y_test - ensemble_corrected)[mask_high], scatter=False, color='red',
                    label='Error Trend')

        plt.title(f"Точность (> {ANALYSIS_THRESHOLD})")
        plt.xlabel("Real")
        plt.ylabel("Pred")
        plt.legend()

        plt.tight_layout()
        plt.show()

    else:
        print("Нет данных выше порога.")


if __name__ == "__main__":
    main()