import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
import warnings

warnings.filterwarnings('ignore')

# --- НАСТРОЙКИ ---
INPUT_FILE = 'million.csv'
MODEL_FILENAME = "final_model.joblib"
RANDOM_STATE = 42
RANGE_MIN = 0.80
RANGE_MAX = 1.00


# --- CLASSES (COPY-PASTE) ---
# (Оставляем те же классы, что и раньше, чтобы joblib работал)
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
    def __init__(self, p, lq, lm, xq, xm, cq, cm):
        self.preprocessor = p
        self.bias_corrector = IsotonicCorrector()
        self.base_models = []
        self.final_estimator = Ridge(alpha=0.5, random_state=RANDOM_STATE)

    def predict_logit_raw(self, X_raw_or_transformed):
        if isinstance(X_raw_or_transformed, pd.DataFrame) and 'cluster_id' in X_raw_or_transformed.columns:
            X = X_raw_or_transformed
        else:
            X = self.preprocessor.transform(X_raw_or_transformed)
        meta_features = np.column_stack([model.predict(X) for name, model in self.base_models])
        return self.final_estimator.predict(meta_features)

    def predict(self, X_raw, apply_calibration=True):
        logits = self.predict_logit_raw(X_raw)
        preds = from_logit(logits)
        if apply_calibration: preds = self.bias_corrector.predict(preds)
        return preds


# Mock modules for joblib
import types

module_fake = types.ModuleType('ultimate_stable')
module_fake.SmartPreprocessor = SmartPreprocessor
module_fake.IsotonicCorrector = IsotonicCorrector
module_fake.HybridModelGod = HybridModelGod
module_fake.to_logit = to_logit
module_fake.from_logit = from_logit
sys.modules['ultimate_stable'] = module_fake


# --- НОВЫЕ МЕТОДЫ КОРРЕКЦИИ ---

def apply_offset_correction(y_pred, y_real_mean):
    # Просто сдвигаем среднее к реальному, не меняя разброс
    bias = y_real_mean - np.mean(y_pred)
    return np.clip(y_pred + bias, 0, 1)


def apply_minmax_stretch(y_pred, target_min, target_max):
    # Растягиваем: (x - min) / (max - min) * (new_max - new_min) + new_min
    p_min, p_max = np.percentile(y_pred, 1), np.percentile(y_pred, 99.9)  # берем перцентили, чтобы выбросы не ломали

    # Нормализуем в 0..1
    y_norm = (y_pred - p_min) / (p_max - p_min)

    # Масштабируем в целевой диапазон (например, 0.80 - 1.00)
    # Но лучше масштабировать в реальный диапазон выборки
    y_stretched = y_norm * (target_max - target_min) + target_min
    return np.clip(y_stretched, 0, 1)


def apply_quantile_mapping(y_pred, y_real_distribution):
    # Сортируем оба массива
    # Это "идеальное" преобразование распределения
    y_pred_sorted = np.sort(y_pred)
    y_real_sorted = np.sort(y_real_distribution)

    # Создаем mapping функцию (интерполяцию)
    # Мы говорим: "если модель дала X, то это соответствует Y в реальном распределении"
    return np.interp(y_pred, y_pred_sorted, y_real_sorted)


# --- MAIN ---
def main():
    print(f"=== BATTLE OF CORRECTORS ({RANGE_MIN}-{RANGE_MAX}) ===\n")

    # 1. Load Data & Model
    if not os.path.exists(INPUT_FILE): return
    df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)
    X = df.drop(columns=['target'])
    y = df['target'].values
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Filter
    mask_range = (y_test >= RANGE_MIN) & (y_test <= RANGE_MAX)
    X_target = X_test[mask_range]
    y_target = y_test[mask_range]
    print(f"🎯 Точек: {len(y_target)}")

    model = joblib.load(MODEL_FILENAME)
    # Patch if needed
    if not hasattr(model, 'predict_logit_raw'):
        from types import MethodType
        def patched_predict_logit_raw(self, X_raw_or_transformed):
            if isinstance(X_raw_or_transformed, pd.DataFrame) and 'cluster_id' in X_raw_or_transformed.columns:
                X = X_raw_or_transformed
            else:
                X = self.preprocessor.transform(X_raw_or_transformed)
            meta_features = np.column_stack([model.predict(X) for name, model in self.base_models])
            return self.final_estimator.predict(meta_features)

        model.predict_logit_raw = MethodType(patched_predict_logit_raw, model)

    # Get Raw Preds
    logits_raw = model.predict_logit_raw(X_target)
    preds_raw = from_logit(logits_raw)

    # --- МЕТОДЫ ---

    # 0. Старая Линейка (Sklearn)
    lr = LinearRegression()
    lr.fit(preds_raw.reshape(-1, 1), y_target)
    preds_lr = np.clip(lr.predict(preds_raw.reshape(-1, 1)), 0, 1)

    # 1. Offset Only (Простой сдвиг)
    preds_offset = apply_offset_correction(preds_raw, np.mean(y_target))

    # 2. MinMax Stretch (Растягивание)
    # Растягиваем предсказания так, чтобы они покрывали диапазон от Min(Real) до Max(Real)
    preds_stretch = apply_minmax_stretch(preds_raw, np.min(y_target), np.max(y_target))

    # 3. Quantile Mapping (Подмена распределения)
    preds_quant = apply_quantile_mapping(preds_raw, y_target)

    # --- АНАЛИЗ ---
    results = {
        "Raw Stack": preds_raw,
        "Old Linear": preds_lr,
        "Offset Only": preds_offset,
        "MinMax Stretch": preds_stretch,
        "Quantile Map": preds_quant
    }

    print(f"\n{'Method':<20} | {'RMSE':<10} | {'Bias':<10} | {'Max Pred':<10}")
    print("-" * 60)
    for name, pred in results.items():
        rmse = np.sqrt(mean_squared_error(y_target, pred))
        bias = np.mean(pred - y_target)
        max_p = np.max(pred)
        print(f"{name:<20} | {rmse:.5f}    | {bias:.5f}    | {max_p:.5f}")

    # --- ГРАФИКИ ---
    plt.figure(figsize=(18, 6))

    # 1. Scatter Plots
    plt.subplot(1, 3, 1)
    plt.scatter(y_target, preds_lr, alpha=0.1, s=5, c='gray', label='Old Linear')
    plt.scatter(y_target, preds_stretch, alpha=0.2, s=5, c='orange', label='Stretched')
    plt.plot([RANGE_MIN, 1], [RANGE_MIN, 1], 'k--', lw=2)
    plt.title("Linear vs Stretched")
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.scatter(y_target, preds_quant, alpha=0.2, s=5, c='crimson', label='Quantile Map')
    plt.plot([RANGE_MIN, 1], [RANGE_MIN, 1], 'k--', lw=2)
    plt.title("Quantile Mapping (Ideal Dist)")
    plt.legend()

    # 2. Histograms
    plt.subplot(1, 3, 3)
    sns.kdeplot(y_target, label='REAL Target', color='black', linewidth=3)
    sns.kdeplot(preds_lr, label='Old Linear', color='gray', linestyle='--')
    sns.kdeplot(preds_stretch, label='Stretched', color='orange', fill=True, alpha=0.3)
    sns.kdeplot(preds_quant, label='Quantile', color='crimson', fill=True, alpha=0.2)
    plt.title("Distribution Shapes")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()