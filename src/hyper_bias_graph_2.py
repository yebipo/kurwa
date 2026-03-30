import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

# --- 1. НАСТРОЙКИ ---

INPUT_FILE = 'million.csv'
MODEL_FILENAME = "final_model_stable.joblib"
# Сюда сохраним соло-корректор
SOLO_CORRECTOR_PATH = "xgb_q_linear_corrector.joblib"

RANDOM_STATE = 42
EPSILON = 1e-5
QUANTILE_ALPHA = 0.90


# --- 2. КЛАССЫ (ОБЯЗАТЕЛЬНО) ---
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
    def __init__(self, p, lq, lm, xq, xm, cq,
                 cm): self.preprocessor = p; self.bias_corrector = IsotonicCorrector(); self.model = None

    def predict(self, X): pass


# --- 3. MAIN ---
def main():
    print("=== ЭКСПЕРИМЕНТ: SOLO XGB_Q + LINEAR CORRECTION ===\n")

    # 1. Загрузка данных
    if not os.path.exists(INPUT_FILE): return
    print("📂 Читаем данные...")
    df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)

    X = df.drop(columns=['target'])
    y = df['target']

    # Делим так же, как при обучении, чтобы проверить на Тесте
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # 2. Загрузка God Model
    print("🔄 Загрузка God Model (чтобы достать XGB)...")
    try:
        god_model = joblib.load(MODEL_FILENAME)
        print("   ✅ Загружено.")
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return

    # 3. Извлечение XGB_Q
    print("🕵️‍♂️ Извлекаем XGBoost Quantile...")
    xgb_q_model = None
    for name, model in god_model.base_models:
        if name == 'xgb_q':
            xgb_q_model = model
            break

    if xgb_q_model is None:
        print("❌ XGB_Q не найден внутри ансамбля!")
        return
    print("   ✅ XGB_Q найден.")

    # 4. Препроцессинг и Предсказание
    print("🧠 Предсказание (XGB Raw)...")
    # Используем препроцессор из God Model, так как XGB учился на нем
    X_test_transformed = god_model.preprocessor.transform(X_test)
    preds_raw = xgb_q_model.predict(X_test_transformed)

    # 5. Обучение Линейного Корректора
    print("🔧 Обучение Линейной Коррекции...")
    # Bias = Real - Pred
    residuals = y_test - preds_raw

    corrector = LinearRegression()
    # Учимся предсказывать ошибку по значению прогноза
    corrector.fit(preds_raw.reshape(-1, 1), residuals)

    bias_pred = corrector.predict(preds_raw.reshape(-1, 1))

    # Применяем поправку
    preds_corrected = preds_raw + bias_pred
    preds_corrected = np.clip(preds_corrected, 0, 1)

    # Сохраняем корректор
    joblib.dump(corrector, SOLO_CORRECTOR_PATH)
    print(f"   Формула: Correction = {corrector.coef_[0]:.4f} * Pred + {corrector.intercept_:.4f}")

    # 6. Оценка (Фокус на высоких > 0.9)
    mask_high = y_test > 0.90

    rmse_raw = np.sqrt(mean_squared_error(y_test[mask_high], preds_raw[mask_high]))
    rmse_cor = np.sqrt(mean_squared_error(y_test[mask_high], preds_corrected[mask_high]))

    r2_raw = r2_score(y_test[mask_high], preds_raw[mask_high])
    r2_cor = r2_score(y_test[mask_high], preds_corrected[mask_high])

    bias_raw = np.mean(y_test[mask_high] - preds_raw[mask_high])
    bias_cor = np.mean(y_test[mask_high] - preds_corrected[mask_high])

    print("\n" + "=" * 60)
    print(f"{'METRIC (Real > 0.90)':<25} | {'RAW XGB_Q':<15} | {'CORRECTED':<15}")
    print("=" * 60)
    print(f"{'RMSE':<25} | {rmse_raw:.5f}         | {rmse_cor:.5f}")
    print(f"{'R2 Score':<25} | {r2_raw:.5f}         | {r2_cor:.5f}")
    print(f"{'Mean Bias':<25} | {bias_raw:.5f}         | {bias_cor:.5f}")
    print("-" * 60)

    # 7. Графики
    plt.figure(figsize=(14, 6))

    # Scatter на высоких
    plt.subplot(1, 2, 1)
    plt.scatter(y_test[mask_high], preds_raw[mask_high], alpha=0.2, s=5, c='gray', label='Raw XGB_Q')
    plt.scatter(y_test[mask_high], preds_corrected[mask_high], alpha=0.2, s=5, c='red', label='Corrected')
    plt.plot([0.9, 1], [0.9, 1], 'k--', lw=2)
    plt.title("XGB_Q Solo: Точность на пиках")
    plt.xlabel("Real")
    plt.ylabel("Pred")
    plt.legend()

    # Bias Distribution
    plt.subplot(1, 2, 2)
    sns.histplot(y_test[mask_high] - preds_corrected[mask_high], bins=50, kde=True, color='green')
    plt.axvline(0, color='red', linestyle='--')
    plt.title("Распределение ошибки (Corrected, >0.9)")
    plt.xlabel("Error")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()