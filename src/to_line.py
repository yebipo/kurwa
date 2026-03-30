import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler, PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error  # <--- ВОТ ЧЕГО НЕ ХВАТАЛО
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

# --- 1. НАСТРОЙКИ ---
WORK_DIR = "."
INPUT_FILE = 'million.csv'
MODEL_FILE = "final_model_unbiased.joblib"
GLOBAL_CORRECTOR_PATH = "global_error_function.joblib"

RANDOM_STATE = 42
EPSILON = 1e-5


# --- 2. КЛАССЫ (НУЖНЫ ДЛЯ ЗАГРУЗКИ JOBLIB) ---
def to_logit(y): return np.log(np.clip(y, EPSILON, 1 - EPSILON) / (1 - np.clip(y, EPSILON, 1 - EPSILON)))


def from_logit(z): return 1 / (1 + np.exp(-z))


# A. Standard Classes
class GlobalPreprocessor:
    def __init__(self, scaler, kmeans): self.scaler = scaler; self.kmeans = kmeans

    def transform(self, X):
        l = np.log2(X.astype(float).clip(lower=1));
        s = self.scaler.transform(l);
        o = l.copy();
        o['cluster_id'] = self.kmeans.predict(s);
        return o


class ProductionModel:
    def __init__(self, p, l, x, c): self.preprocessor = p; self.estimators = [('lgbm', l), ('xgb', x),
                                                                              ('catboost', c)]; self.model = None

    def predict(self, X): return self.model.predict(self.preprocessor.transform(X))


# B. God Mode Classes
class SmartPreprocessor:
    def __init__(self, n_clusters=20):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
        self.n_clusters = n_clusters

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1));
        X_scaled = self.scaler.transform(X_log)
        df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)
        df['row_mean'] = df.mean(axis=1);
        df['row_std'] = df.std(axis=1);
        df['row_max'] = df.max(axis=1)
        dists = self.kmeans.transform(X_scaled)
        for i in range(self.n_clusters): df[f'dist_clust_{i}'] = dists[:, i]
        df['cluster_id'] = self.kmeans.predict(X_scaled)
        return df

    def fit(self, X): pass


class IsotonicCorrector:
    def __init__(self): self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    def predict(self, y): return self.iso.predict(y)


class HybridModelGod:
    def __init__(self, p, lq, lm, xq, xm, cq, cm):
        self.preprocessor = p;
        self.bias_corrector = IsotonicCorrector();
        self.model = None;
        self.base_models = []

    def predict(self, X, apply_calibration=True):
        t = self.preprocessor.transform(X);
        mf = np.column_stack([m.predict(t) for n, m in self.base_models])
        l = self.final_estimator.predict(mf);
        p = from_logit(l)
        return self.bias_corrector.predict(p) if apply_calibration else p


# C. Wrappers
class CalibratedProductionModel:
    def __init__(self, base_model, calibrator): self.base_model = base_model; self.calibrator = calibrator

    def predict(self, X_raw): return self.calibrator.predict(self.base_model.predict(X_raw).reshape(-1, 1))


class AdditiveCalibratedModel:
    def __init__(self, base_model, bias_value): self.base_model = base_model; self.bias_value = float(bias_value)

    def predict(self, X_raw): return self.base_model.predict(X_raw) + self.bias_value


# D. Stubs
class FinalQuantileWrapper:
    def __init__(self, model=None): self.model = model

    def predict(self, X, **kwargs): return self.model.predict(X)


class ProductionModelLogit(ProductionModel): pass


class HighEndQuantileMapper:
    def __init__(self, base_model=None): self.base_model = base_model

    def predict(self, X, **kwargs): return self.base_model.predict(X)


class LogitBoosterWrapper:
    def __init__(self, model=None): self.model = model

    def predict(self, X, **kwargs): return self.model.predict(X)


class FinalModel:
    def __init__(self, lgbm_params, xgb_params, catboost_params, n_clusters, random_state=42):
        self.n_clusters = n_clusters;
        self.scaler = StandardScaler();
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.stacking_model = StackingRegressor(estimators=[], final_estimator=LinearRegression())

    def predict(self, X_raw): return np.zeros(len(X_raw))


# --- 3. BATCH PREDICT ---
def predict_batch(model, X_df):
    preds = []
    batch_size = 50000
    use_calib = False
    try:
        model.predict(X_df.iloc[:2], apply_calibration=True)
        use_calib = True
    except:
        use_calib = False

    print("   -> Predicting...", end=" ")
    for i in range(0, len(X_df), batch_size):
        if i % 200000 == 0: print(".", end="", flush=True)
        batch = X_df.iloc[i:i + batch_size]
        try:
            if use_calib:
                p = model.predict(batch, apply_calibration=True)
            else:
                p = model.predict(batch)
            if p.ndim > 1: p = p.flatten()
            preds.append(p)
        except Exception as e:
            print(f"[ERR: {e}]")
            return None
    print(" Done.")
    return np.hstack(preds)


# --- 4. MAIN ---
def main():
    print(f"=== LINEAR ERROR CORRECTION (Slope Adjustment) ===\n")

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
    y_real = df['target'].values

    print(f"🔄 Загрузка {MODEL_FILE}...")
    try:
        model = joblib.load(MODEL_FILE)
        print("   ✅ Успешно.")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    # 1. Получаем предсказания
    y_pred = predict_batch(model, X)
    if y_pred is None: return

    # 2. Считаем ошибку (Error = Real - Pred)
    error = y_real - y_pred

    print("\n📏 Обучаем линейную коррекцию (Error ~ Pred)...")

    # Строим зависимость Ошибки от Предсказания
    lin_corrector = LinearRegression()
    lin_corrector.fit(y_pred.reshape(-1, 1), error)

    slope = lin_corrector.coef_[0]
    intercept = lin_corrector.intercept_
    print(f"   [Формула] Correction = {slope:.5f} * Pred + {intercept:.5f}")

    # 3. Применяем
    predicted_bias = lin_corrector.predict(y_pred.reshape(-1, 1))
    y_corrected = y_pred + predicted_bias
    y_corrected = np.clip(y_corrected, 0, 1)

    # Сохраняем
    joblib.dump(lin_corrector, GLOBAL_CORRECTOR_PATH)
    print(f"💾 Корректор сохранен в {GLOBAL_CORRECTOR_PATH}")

    # 4. Метрики
    rmse_orig = np.sqrt(mean_squared_error(y_real, y_pred))
    rmse_corr = np.sqrt(mean_squared_error(y_real, y_corrected))

    mask_high = y_real > 0.90
    if np.sum(mask_high) > 0:
        rmse_high_orig = np.sqrt(mean_squared_error(y_real[mask_high], y_pred[mask_high]))
        rmse_high_corr = np.sqrt(mean_squared_error(y_real[mask_high], y_corrected[mask_high]))
    else:
        rmse_high_orig, rmse_high_corr = 0, 0

    print("\n" + "=" * 55)
    print(f"{'METRIC':<25} | {'ORIGINAL':<10} | {'CORRECTED':<10}")
    print("-" * 55)
    print(f"{'Global RMSE':<25} | {rmse_orig:.5f}    | {rmse_corr:.5f}")
    print(f"{'High RMSE (>0.9)':<25} | {rmse_high_orig:.5f}    | {rmse_high_corr:.5f}")
    print("-" * 55)

    # 5. Графики
    plt.figure(figsize=(18, 6))
    idx = np.random.choice(len(y_real), size=min(50000, len(y_real)), replace=False)

    # График 1: Error vs Pred (Обучающая выборка для корректора)
    plt.subplot(1, 3, 1)
    plt.scatter(y_pred[idx], error[idx], alpha=0.1, s=1, c='gray')
    x_range = np.linspace(0, 1, 100).reshape(-1, 1)
    plt.plot(x_range, lin_corrector.predict(x_range), 'r-', lw=3, label='Correction Line')
    plt.axhline(0, color='black', linestyle='--')
    plt.title(f"Fit: Error vs Pred\nSlope: {slope:.3f}")
    plt.xlabel("Pred")
    plt.ylabel("Error")
    plt.legend()

    # График 2: Error vs Real (Проверка тренда)
    plt.subplot(1, 3, 2)
    plt.scatter(y_real[idx], error[idx], alpha=0.1, s=1, c='green', label='Original Error')
    plt.scatter(y_real[idx], predicted_bias[idx], alpha=0.1, s=1, c='red', label='Applied Correction')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Visual Check: Error vs Real")
    plt.xlabel("Real")
    plt.legend()

    # График 3: Результат
    plt.subplot(1, 3, 3)
    plt.scatter(y_real[idx], y_pred[idx], alpha=0.1, s=1, c='blue', label='Original')
    plt.scatter(y_real[idx], y_corrected[idx], alpha=0.1, s=1, c='orange', label='Corrected')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title("Pred vs Real")
    plt.xlabel("Real")
    plt.ylabel("Pred")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()