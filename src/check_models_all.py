import pandas as pd
import numpy as np
import joblib
import os
import glob
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. НАСТРОЙКИ
# ==============================================================================
WORK_DIR = "."
INPUT_FILE = 'million.csv'
PLOTS_DIR = "benchmark_plots"
HIGH_VALUE_THRESHOLD = 0.90
BATCH_SIZE = 50000

RANDOM_STATE = 42
EPSILON = 1e-5

if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)


# ==============================================================================
# 2. КЛАССЫ
# ==============================================================================

def to_logit(y): return np.log(np.clip(y, EPSILON, 1 - EPSILON) / (1 - np.clip(y, EPSILON, 1 - EPSILON)))


def from_logit(z): return 1 / (1 + np.exp(-z))


# --- A. Old Model Classes ---
class GlobalPreprocessor:
    def __init__(self, scaler, kmeans): self.scaler = scaler; self.kmeans = kmeans

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        X_out = X_log.copy()
        X_out['cluster_id'] = self.kmeans.predict(X_scaled)
        return X_out


class ProductionModel:
    def __init__(self, preprocessor, lgbm_p, xgb_p, cat_p):
        self.preprocessor = p;
        self.estimators = [('lgbm', l), ('xgb', x), ('catboost', c)];
        self.model = None

    def fit(self, X, y): pass

    def predict(self, X_raw): return self.model.predict(self.preprocessor.transform(X_raw))


# --- B. God Mode Classes ---
class SmartPreprocessor:
    def __init__(self, n_clusters=20):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
        self.n_clusters = n_clusters

    def fit(self, X): pass

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


class IsotonicCorrector:
    def __init__(self): self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    def fit(self, y, t): pass

    def predict(self, y): return self.iso.predict(y)


class HybridModelGod:
    def __init__(self, p, lq, lm, xq, xm, cq, cm):
        self.preprocessor = p;
        self.bias_corrector = IsotonicCorrector();
        self.model = None;
        self.base_models = [];
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


# --- C. Wrappers & Stubs ---
class CalibratedProductionModel:
    def __init__(self, base_model, calibrator): self.base_model = base_model; self.calibrator = calibrator

    def predict(self, X_raw): return self.calibrator.predict(self.base_model.predict(X_raw).reshape(-1, 1))


class AdditiveCalibratedModel:
    def __init__(self, base_model, bias_value): self.base_model = base_model; self.bias_value = float(bias_value)

    def predict(self, X_raw): return self.base_model.predict(X_raw) + self.bias_value


class FinalModel:
    def __init__(self, lgbm_params, xgb_params, catboost_params, n_clusters, random_state=42):
        self.n_clusters = n_clusters;
        self.scaler = StandardScaler();
        self.kmeans = KMeans(n_clusters=n_clusters)
        self.stacking_model = StackingRegressor(estimators=[], final_estimator=LinearRegression())

    def _log_transform(self, X):
        X_t = X.copy()
        for col in X_t.columns: X_t[col] = np.log2(X_t[col].astype(float).clip(lower=1))
        return X_t

    def _add_cluster_features(self, X_log, is_fit=False):
        X_s = self.scaler.transform(X_log);
        cl = self.kmeans.predict(X_s)
        X_c = X_log.copy();
        X_c['cluster_id'] = cl
        return X_c

    def predict(self, X_raw):
        # FIX: Убрана проверка имен колонок
        X_log = self._log_transform(X_raw)
        X_final = self._add_cluster_features(X_log, is_fit=False)
        return self.stacking_model.predict(X_final)


class FinalQuantileWrapper:
    def __init__(self, model=None): self.model = model

    def predict(self, X, **kwargs): return self.model.predict(X) if hasattr(self, 'model') else np.zeros(len(X))


class ProductionModelLogit(ProductionModel): pass


class HighEndQuantileMapper:
    def __init__(self, base_model=None): self.base_model = base_model

    def predict(self, X, **kwargs): return self.base_model.predict(X)


class LogitBoosterWrapper:
    def __init__(self, model=None): self.model = model

    def predict(self, X, **kwargs): return self.model.predict(X) if hasattr(self, 'model') else np.zeros(len(X))


# ==============================================================================
# 3. УМНЫЙ BATCH PREDICT (AUTO-FIX COLUMNS)
# ==============================================================================
def get_compatible_batch(X_batch, strategy):
    """Преобразует батч данных под требования модели"""
    if strategy == 'original':
        return X_batch
    elif strategy == 'rename_digits':
        # Для model.joblib (1, 2, 3, 4)
        X_new = X_batch.copy()
        X_new.columns = ['1', '2', '3', '4']
        return X_new
    elif strategy == 'subset_1_3':
        # Для model_log_2par.joblib (только feature_1 и feature_3)
        return X_batch[['feature_1', 'feature_3']]
    return X_batch


def predict_batch_smart(model, X_full):
    """
    1. Определяет рабочую стратегию данных на маленьком куске.
    2. Применяет её ко всем данным батчами.
    """
    sample = X_full.iloc[:5]
    best_strategy = None
    use_calib = False

    # 1. Подбор стратегии
    strategies = ['original', 'rename_digits', 'subset_1_3']

    for strat in strategies:
        try:
            X_test = get_compatible_batch(sample, strat)

            # Пробуем с калибровкой
            try:
                model.predict(X_test, apply_calibration=True)
                use_calib = True
                best_strategy = strat
                break
            except TypeError:
                # Пробуем без калибровки
                model.predict(X_test)
                use_calib = False
                best_strategy = strat
                break
            except Exception:
                # Если упало внутри predict (например feature mismatch), идем к след стратегии
                pass
        except Exception:
            continue

    if best_strategy is None:
        print(" [Несовместимый формат данных]", end="")
        return None

    # 2. Основной цикл
    preds = []
    n_rows = len(X_full)

    for i in range(0, n_rows, BATCH_SIZE):
        batch_raw = X_full.iloc[i: i + BATCH_SIZE]
        batch_ready = get_compatible_batch(batch_raw, best_strategy)

        try:
            if use_calib:
                p = model.predict(batch_ready, apply_calibration=True)
            else:
                p = model.predict(batch_ready)

            if p.ndim > 1: p = p.flatten()
            preds.append(p)
        except Exception as e:
            print(f" [Err batch {i}: {e}]", end="")
            return None

    return np.hstack(preds)


# ==============================================================================
# 4. ОТРИСОВКА
# ==============================================================================
def plot_model_performance(y_true, y_pred, model_name, rmse_all, bias_all):
    plt.figure(figsize=(18, 5))

    # 1. Scatter
    plt.subplot(1, 3, 1)
    idx = np.random.choice(len(y_true), size=min(20000, len(y_true)), replace=False)
    plt.scatter(y_true[idx], y_pred[idx], alpha=0.1, s=2, c='blue')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title(f"{model_name}\nRMSE: {rmse_all:.4f}")
    plt.xlabel("Real");
    plt.ylabel("Pred");
    plt.xlim(0, 1);
    plt.ylim(0, 1)

    # 2. Bias Hist
    plt.subplot(1, 3, 2)
    bias = y_true - y_pred
    sns.histplot(bias, bins=100, kde=True, color='purple')
    plt.axvline(0, color='red', linestyle='--', lw=2)
    plt.title(f"Bias Distribution\nMean: {bias_all:.5f}")
    plt.xlim(-0.5, 0.5)

    # 3. Error vs Real
    plt.subplot(1, 3, 3)
    plt.scatter(y_true[idx], bias[idx], alpha=0.1, s=2, c='green')
    plt.axhline(0, color='black', lw=2)
    mask_high = y_true > HIGH_VALUE_THRESHOLD
    if mask_high.sum() > 0:
        sns.regplot(x=y_true[mask_high], y=bias[mask_high], scatter=False, color='red', label='Trend > 0.9')
        plt.legend()
    plt.title("Error vs Real Value")
    plt.ylim(-0.5, 0.5)

    plt.tight_layout()
    filename = f"{PLOTS_DIR}/plot_{model_name}.png"
    plt.savefig(filename)
    plt.close()


# ==============================================================================
# 5. MAIN
# ==============================================================================
def main():
    print(f"=== БЕНЧМАРК (SMART DATA FIX) ===\n")
    if not os.path.exists(INPUT_FILE): return

    print("📂 Читаем данные...")
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                         usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
        df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df.dropna(subset=['target'], inplace=True)
        X = df.drop(columns=['target'])
        y_real = df['target'].values
        print(f"   Строк: {len(X)}")
    except Exception as e:
        return

    files = glob.glob("*.joblib") + glob.glob("*.pkl")
    results = []

    for fname in files:
        if any(x in fname for x in ["corrector", "scaler", "preprocessor"]): continue
        print(f"\n👉 {fname} ...", end=" ", flush=True)

        try:
            obj = joblib.load(fname)
            if not hasattr(obj, 'predict'):
                print("⚠️ Не модель.")
                continue

            # --- УМНЫЙ ПРЕДИКШН ---
            preds = predict_batch_smart(obj, X)

            if preds is None:
                print("❌ Fail.")
                continue

            # Метрики
            rmse_all = np.sqrt(mean_squared_error(y_real, preds))
            bias_all = np.mean(y_real - preds)
            mask_high = y_real > HIGH_VALUE_THRESHOLD
            if mask_high.sum() > 0:
                bias_high = np.mean(y_real[mask_high] - preds[mask_high])
                rmse_high = np.sqrt(mean_squared_error(y_real[mask_high], preds[mask_high]))
            else:
                bias_high = np.nan; rmse_high = np.nan

            print("✅ OK -> 📸", end=" ")
            plot_model_performance(y_real, preds, fname, rmse_all, bias_all)

            results.append({'Model': fname, 'RMSE(All)': rmse_all, 'Bias(All)': bias_all, 'RMSE(>0.9)': rmse_high,
                            'Bias(>0.9)': bias_high})

        except Exception as e:
            print(f"❌ Crash: {e}")

    if results:
        res_df = pd.DataFrame(results).sort_values(by='RMSE(>0.9)', ascending=True)
        print("\n" + "=" * 95)
        print("🏆 ИТОГОВЫЙ РЕЙТИНГ")
        print("=" * 95)
        print(res_df.to_string(index=False, float_format=lambda x: "{:.5f}".format(x)))
        res_df.to_csv("benchmark_smart_results.csv", index=False)
        print(f"\n📂 Графики сохранены в {PLOTS_DIR}")


if __name__ == "__main__":
    main()