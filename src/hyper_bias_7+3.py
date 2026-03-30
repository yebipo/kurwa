import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import r2_score, mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')


INPUT_FILE = 'million.csv'

OLD_MODEL_PATH = "final_model.joblib"
NEW_MODEL_PATH = "final_model_stable.joblib"

RANDOM_STATE = 42
EPSILON = 1e-5
QUANTILE_ALPHA = 0.90


# --- 2. КЛАССЫ (ОБЯЗАТЕЛЬНО) ---
def to_logit(y): return np.log(np.clip(y, 1e-5, 1 - 1e-5) / (1 - np.clip(y, 1e-5, 1 - 1e-5)))


def from_logit(z): return 1 / (1 + np.exp(-z))


# Old Model Classes
class GlobalPreprocessor:
    def __init__(self, scaler, kmeans): self.scaler = scaler; self.kmeans = k

    def transform(self, X):
        l = np.log2(X.astype(float).clip(lower=1));
        s = self.scaler.transform(l);
        o = l.copy();
        o['cluster_id'] = self.kmeans.predict(s);
        return o


class ProductionModel:
    def __init__(self, p, l, x, c): self.preprocessor = p; self.estimators = [('lgbm', l), ('xgb', x),
                                                                              ('catboost', c)]; self.model = None

    def predict(self, X): pass


# New Model Classes
class SmartPreprocessor:
    def __init__(self, n_clusters=20):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
        self.n_clusters = n_clusters

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

    def predict(self, X): pass


# --- 3. ФУНКЦИИ АНАЛИЗА ---

def predict_in_batches(model, X, batch_size=50000):
    """
    Безопасное предсказание кусками (чтобы не переполнить RAM).
    Особенно важно для MLPRegressor на больших данных.
    """
    n_samples = X.shape[0]
    preds_list = []

    # Если X это DataFrame
    is_df = hasattr(X, 'iloc')

    for i in range(0, n_samples, batch_size):
        end = min(i + batch_size, n_samples)
        if is_df:
            X_batch = X.iloc[i:end]
        else:
            X_batch = X[i:end]

        p = model.predict(X_batch)
        preds_list.append(p)

    return np.hstack(preds_list)


def analyze_submodel(name, model, X_transformed, y_real, ax_hist, ax_scat):
    # 1. Прогноз (ИСПОЛЬЗУЕМ БАТЧИ!)
    # Здесь мы передаем X_transformed, который уже в памяти,
    # но модель (особенно MLP) внутри может создавать огромные матрицы.
    # Batching спасает RAM.
    preds = predict_in_batches(model, X_transformed, batch_size=50000)

    # 2. Коррекция
    residuals = y_real - preds
    corrector = LinearRegression()
    # Учимся на всем объеме (LinearRegression ест мало памяти)
    corrector.fit(preds.reshape(-1, 1), residuals)

    bias_pred = corrector.predict(preds.reshape(-1, 1))
    preds_corrected = np.clip(preds + bias_pred, 0, 1)

    # 3. Метрики
    bias_before = np.mean(residuals)
    bias_after = np.mean(y_real - preds_corrected)

    # 4. График Гистограммы
    # Для графика берем сэмпл, чтобы не тормозило отрисовку
    sample_idx = np.random.choice(len(y_real), size=min(50000, len(y_real)), replace=False)

    sns.histplot((y_real - preds)[sample_idx], bins=50, color='gray', alpha=0.3, label='Before', kde=True, ax=ax_hist)
    sns.histplot((y_real - preds_corrected)[sample_idx], bins=50, color='blue', alpha=0.3, label='After', kde=True,
                 ax=ax_hist)
    ax_hist.axvline(0, color='red', linestyle='--')
    ax_hist.set_title(f"{name}\nBias: {bias_before:.3f} -> {bias_after:.3f}")
    ax_hist.legend(fontsize='small')

    # 5. График Наклона (Scatter)
    idx_scatter = np.random.choice(len(y_real), size=min(2000, len(y_real)), replace=False)
    ax_scat.scatter(y_real[idx_scatter], preds[idx_scatter], alpha=0.1, s=1, c='gray')
    ax_scat.scatter(y_real[idx_scatter], preds_corrected[idx_scatter], alpha=0.1, s=1, c='red')
    ax_scat.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_scat.set_title(f"Slope: {corrector.coef_[0]:.2f}")

    return preds, preds_corrected


# --- 4. MAIN ---
def main():
    print("=== ГЛУБОКИЙ АНАЛИЗ (10 МОДЕЛЕЙ) [Safe RAM Mode] ===\n")

    # Загрузка данных
    if not os.path.exists(INPUT_FILE):
        print("❌ Файл CSV не найден.")
        return
    print("📂 Читаем данные...")
    df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)
    X = df.drop(columns=['target'])
    y_real = df['target'].values

    # Загрузка
    print("🔄 Загрузка моделей...")
    try:
        old_wrapper = joblib.load(OLD_MODEL_PATH)
        new_wrapper = joblib.load(NEW_MODEL_PATH)
        print("   ✅ Модели загружены.")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    # --- СБОР И АНАЛИЗ ---
    all_raw_preds = []
    all_corrected_preds = []

    fig, axes = plt.subplots(5, 4, figsize=(20, 20))
    axes_flat = axes.flatten()
    model_axes = [(axes_flat[i * 2], axes_flat[i * 2 + 1]) for i in range(10)]
    fig.suptitle("Коррекция каждого компонента отдельно", fontsize=16)

    # 1. СТАРАЯ МОДЕЛЬ (3 шт)
    print("\n⚙️ Обработка Old Models (3)...")
    # Препроцессинг один раз
    X_old = old_wrapper.preprocessor.transform(X)

    for i, ((name, _), trained_model) in enumerate(zip(old_wrapper.estimators, old_wrapper.model.estimators_)):
        print(f"   -> Processing OLD_{name}...")
        # Анализ (без сохранения в финал)
        analyze_submodel(f"OLD_{name}", trained_model, X_old, y_real, model_axes[i][0], model_axes[i][1])

    # 2. НОВАЯ МОДЕЛЬ (7 шт)
    print("\n⚙️ Обработка New Models (7)...")
    X_new = new_wrapper.preprocessor.transform(X)

    for i, (name, trained_model) in enumerate(new_wrapper.base_models):
        print(f"   -> Processing NEW_{name}...")
        ax_h, ax_s = model_axes[i + 3]

        p_raw, p_corr = analyze_submodel(f"NEW_{name}", trained_model, X_new, y_real, ax_h, ax_s)

        # Собираем
        all_raw_preds.append(p_raw)
        all_corrected_preds.append(p_corr)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- ФИНАЛЬНЫЙ ГРАФИК ---
    print("\n🎨 Рисуем ИТОГОВЫЙ ГРАФИК...")

    ensemble_corrected = np.mean(all_corrected_preds, axis=0)
    bias_corrected = y_real - ensemble_corrected
    rmse_final = np.sqrt(mean_squared_error(y_real, ensemble_corrected))

    plt.figure(figsize=(12, 6))
    # Рисуем гистограмму на сэмпле (для скорости отрисовки matplotlib)
    sample_idx = np.random.choice(len(bias_corrected), size=min(100000, len(bias_corrected)), replace=False)

    sns.histplot(bias_corrected[sample_idx], bins=100, color='green', alpha=0.6, label=f'Corrected Ensemble', kde=True)
    plt.axvline(0, color='red', linestyle='--', lw=2)
    plt.title(f"ФИНАЛ: Распределение Байаса (7 исправленных моделей)\nRMSE: {rmse_final:.5f}")
    plt.xlabel("Error (Real - Pred)")
    plt.legend()
    plt.show()

    print(f"\n✅ ИТОГОВЫЙ БАЙАС: {bias_corrected.mean():.7f}")
    print(f"✅ ИТОГОВЫЙ RMSE: {rmse_final:.5f}")


if __name__ == "__main__":
    main()