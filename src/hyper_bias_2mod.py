import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

# --- 1. НАСТРОЙКИ ---
INPUT_FILE = 'million.csv'

# Пути к моделям (проверьте свои пути!)
OLD_MODEL_PATH = "final_model.joblib"
NEW_MODEL_PATH = "final_model_stable.joblib"
# Сюда сохраним "выпрямитель" байаса
CORRECTOR_PATH = "linear_bias_corrector.joblib"

# Порог переключения на новую модель
GATE_THRESHOLD = 0.90

RANDOM_STATE = 42
EPSILON = 1e-5
QUANTILE_ALPHA = 0.90


# --- 2. КЛАССЫ ДЛЯ СТАРОЙ МОДЕЛИ ---
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

    def fit(self, X, y): pass

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        return self.model.predict(X_ready)


# --- 3. КЛАССЫ ДЛЯ НОВОЙ МОДЕЛИ ---
def to_logit(y): return np.log(np.clip(y, 1e-5, 1 - 1e-5) / (1 - np.clip(y, 1e-5, 1 - 1e-5)))


def from_logit(z): return 1 / (1 + np.exp(-z))


class SmartPreprocessor:
    def __init__(self, n_clusters=20):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
        self.n_clusters = n_clusters

    def fit(self, X): pass

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)
        df['row_mean'] = df.mean(axis=1);
        df['row_std'] = df.std(axis=1);
        df['row_max'] = df.max(axis=1)
        dists = self.kmeans.transform(X_scaled)
        for i in range(self.n_clusters): df[f'dist_clust_{i}'] = dists[:, i]
        df['cluster_id'] = self.kmeans.predict(X_scaled)
        return df


class IsotonicCorrector:
    def __init__(self): self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    def fit(self, y_p, y_t): pass

    def predict(self, y_p): return self.iso.predict(y_p)


class HybridModelGod:
    def __init__(self, preprocessor, p_lgbm_q, p_lgbm_m, p_xgb_q, p_xgb_m, p_cat_q, p_cat_m):
        self.preprocessor = preprocessor
        self.bias_corrector = IsotonicCorrector()
        self.model = None
        self.base_models = []
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


# --- 4. MAIN ---
def main():
    print(f"\n=== КАСКАД С ЛИНЕЙНОЙ КОРРЕКЦИЕЙ БАЙАСА ===\n")

    # 1. Загрузка данных
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Файл {INPUT_FILE} не найден.")
        return

    print("📂 Читаем данные...")
    df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)

    X = df.drop(columns=['target'])
    y_real = df['target']

    # 2. Загрузка моделей
    print("\n🔄 Загрузка моделей...")
    try:
        old_model = joblib.load(OLD_MODEL_PATH)
        print("   ✅ Old Model загружена.")
        new_model = joblib.load(NEW_MODEL_PATH)
        print("   ✅ New Model (God Mode) загружена.")
    except Exception as e:
        print(f"❌ Ошибка загрузки моделей: {e}")
        return

    # 3. ШАГ 1: Старая модель (Gatekeeper)
    print("\n🚧 Шаг 1: Работает Старая Модель...")
    preds_old = old_model.predict(X)

    # Фильтр
    mask_high = preds_old >= GATE_THRESHOLD
    print(f"   Кандидатов (> {GATE_THRESHOLD}): {mask_high.sum()}")

    # Инициализируем финальный массив прогнозами старой модели
    final_preds = preds_old.copy()

    if mask_high.sum() > 0:
        print("🚧 Шаг 2: God Mode + Обучение Корректора...")

        # Данные для "верхушки"
        X_high = X[mask_high]
        y_high_real = y_real[mask_high]

        # Прогноз God Mode (еще смещенный)
        preds_god_raw = new_model.predict(X_high, apply_calibration=True)

        # --- ОБУЧЕНИЕ ЛИНЕЙНОГО КОРРЕКТОРА ---
        # Мы хотим найти: Bias = a * Pred + b
        # И затем сделать: Corrected = Pred + Bias

        residuals = y_high_real - preds_god_raw

        corrector = LinearRegression()
        corrector.fit(preds_god_raw.reshape(-1, 1), residuals)

        predicted_bias = corrector.predict(preds_god_raw.reshape(-1, 1))

        # Применяем сдвиг
        preds_god_corrected = preds_god_raw + predicted_bias
        preds_god_corrected = np.clip(preds_god_corrected, 0, 1)  # Клиппинг

        # Вставляем в общий массив
        final_preds[mask_high] = preds_god_corrected

        print(f"   🔧 Корректор обучен: Bias = {corrector.coef_[0]:.4f} * Pred + {corrector.intercept_:.4f}")
        joblib.dump(corrector, CORRECTOR_PATH)
        print(f"   💾 Корректор сохранен: {CORRECTOR_PATH}")

    # 4. Анализ Байаса (Только для высоких значений, так как нас интересуют они)
    if mask_high.sum() > 0:
        bias_high_before = y_real[mask_high] - preds_god_raw
        bias_high_after = y_real[mask_high] - final_preds[mask_high]

        print("\n" + "=" * 50)
        print("📊 РЕЗУЛЬТАТЫ КОРРЕКЦИИ (На пиках > 0.9)")
        print("=" * 50)
        print(f"{'Metric':<15} | {'Before Correction':<20} | {'After Correction':<20}")
        print("-" * 60)
        print(f"{'Mean Bias':<15} | {bias_high_before.mean():.6f}             | {bias_high_after.mean():.6f}")
        print(f"{'Median Bias':<15} | {bias_high_before.median():.6f}             | {bias_high_after.median():.6f}")
        print(
            f"{'R2 Score':<15} | {r2_score(y_real[mask_high], preds_god_raw):.4f}             | {r2_score(y_real[mask_high], final_preds[mask_high]):.4f}")

        # 5. Графики
        plt.figure(figsize=(14, 6))

        # График 1: Гистограмма Байаса (До vs После)
        plt.subplot(1, 2, 1)
        sns.histplot(bias_high_before, color='gray', alpha=0.3, label='Before (Shifted)', kde=True)
        sns.histplot(bias_high_after, color='purple', alpha=0.5, label='After (Centered)', kde=True)
        plt.axvline(0, color='red', linestyle='--', lw=2)
        plt.title(f"Распределение Байаса (Цель: пик на 0)\nNew Mean: {bias_high_after.mean():.5f}")
        plt.legend()

        # График 2: Scatter (Выпрямилась ли линия?)
        plt.subplot(1, 2, 2)
        plt.scatter(y_real[mask_high], preds_god_raw, alpha=0.2, s=5, c='gray', label='Before')
        plt.scatter(y_real[mask_high], final_preds[mask_high], alpha=0.2, s=5, c='red', label='After')
        plt.plot([0.8, 1], [0.8, 1], 'k--', lw=2)
        plt.title("Точность на пиках")
        plt.xlabel("Real Entropy")
        plt.ylabel("Predicted Entropy")
        plt.legend()

        plt.tight_layout()
        plt.show()
    else:
        print("⚠️ Не найдено высоких значений для проверки.")


if __name__ == "__main__":
    main()