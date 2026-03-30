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

# --- 1. НАСТРОЙКИ ---
INPUT_FILE = 'million.csv'

# Укажите здесь пути к моделям
OLD_MODEL_PATH = "final_model.joblib"  # Старая модель
NEW_MODEL_PATH = "the_best_model_v6_final.joblib"  # Новая модель (созданная скриптом V6)

RANDOM_STATE = 42
EPSILON = 1e-5


# --- 2. КЛАССЫ (ВСЕ НЕОБХОДИМЫЕ ОПРЕДЕЛЕНИЯ) ---

def to_logit(y):
    return np.log(np.clip(y, 1e-5, 1 - 1e-5) / (1 - np.clip(y, 1e-5, 1 - 1e-5)))


def from_logit(z):
    return 1 / (1 + np.exp(-z))


# --- КЛАССЫ СТАРЫХ МОДЕЛЕЙ ---
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

    def predict(self, X): pass


class SmartPreprocessor:
    def __init__(self, n_clusters=20):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
        self.n_clusters = n_clusters

    def transform(self, X_raw):
        # Заглушка для загрузки
        pass

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


# --- НОВЫЙ КЛАСС (FinalModel из скрипта V6) ---
class FinalModel:
    """
    Класс-обертка для модели V6. Включает всю логику предобработки и предсказания.
    """

    def __init__(self, lgbm_params, xgb_params, catboost_params, n_clusters, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.feature_names_in_ = None

        estimators = [
            ('lgbm', lgb.LGBMRegressor(**lgbm_params)),
            ('xgb', xgb.XGBRegressor(**xgb_params)),
            ('catboost', cb.CatBoostRegressor(**catboost_params))
        ]
        self.stacking_model = StackingRegressor(
            estimators=estimators, final_estimator=LinearRegression(),
            cv=3, n_jobs=-1, verbose=0
        )

    def _log_transform(self, X):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            X_transformed[col] = np.log2(X_transformed[col].astype(float).clip(lower=1))
        return X_transformed

    def _add_cluster_features(self, X_log_transformed, is_fit=False):
        if is_fit:
            X_scaled = self.scaler.fit_transform(X_log_transformed)
            cluster_labels = self.kmeans.fit_predict(X_scaled)
        else:
            X_scaled = self.scaler.transform(X_log_transformed)
            cluster_labels = self.kmeans.predict(X_scaled)

        X_with_cluster = X_log_transformed.copy()
        X_with_cluster['cluster_id'] = cluster_labels
        return X_with_cluster

    def fit(self, X_raw, y):
        self.feature_names_in_ = X_raw.columns.tolist()
        X_log = self._log_transform(X_raw)
        X_final = self._add_cluster_features(X_log, is_fit=True)
        self.stacking_model.fit(X_final, y)
        return self

    def predict(self, X_raw):
        # В режиме анализа проверки имен колонок можно пропустить или адаптировать
        X_log = self._log_transform(X_raw)
        X_final = self._add_cluster_features(X_log, is_fit=False)
        return self.stacking_model.predict(X_final)

    def get_final_features(self, X_raw):
        X_log = self._log_transform(X_raw)
        X_final_features = self._add_cluster_features(X_log, is_fit=False)
        return X_final_features


# --- 3. ФУНКЦИИ АНАЛИЗА ---

def predict_in_batches(model, X, batch_size=50000):
    """
    Безопасное предсказание кусками (чтобы не переполнить RAM).
    """
    n_samples = X.shape[0]
    preds_list = []
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
    # 1. Прогноз
    preds = predict_in_batches(model, X_transformed, batch_size=50000)

    # 2. Коррекция (Linear Regression)
    residuals = y_real - preds
    corrector = LinearRegression()
    corrector.fit(preds.reshape(-1, 1), residuals)

    bias_pred = corrector.predict(preds.reshape(-1, 1))
    preds_corrected = np.clip(preds + bias_pred, 0, 1)

    # 3. Метрики
    bias_before = np.mean(residuals)
    bias_after = np.mean(y_real - preds_corrected)

    # 4. Графики
    sample_idx = np.random.choice(len(y_real), size=min(50000, len(y_real)), replace=False)

    sns.histplot((y_real - preds)[sample_idx], bins=50, color='gray', alpha=0.3, label='Before', kde=True, ax=ax_hist)
    sns.histplot((y_real - preds_corrected)[sample_idx], bins=50, color='blue', alpha=0.3, label='After', kde=True,
                 ax=ax_hist)
    ax_hist.axvline(0, color='red', linestyle='--')
    ax_hist.set_title(f"{name}\nBias: {bias_before:.3f} -> {bias_after:.3f}")
    ax_hist.legend(fontsize='small')

    idx_scatter = np.random.choice(len(y_real), size=min(2000, len(y_real)), replace=False)
    ax_scat.scatter(y_real[idx_scatter], preds[idx_scatter], alpha=0.1, s=1, c='gray')
    ax_scat.scatter(y_real[idx_scatter], preds_corrected[idx_scatter], alpha=0.1, s=1, c='red')
    ax_scat.plot([0, 1], [0, 1], 'k--', lw=1)
    ax_scat.set_title(f"Slope: {corrector.coef_[0]:.2f}")

    return preds, preds_corrected


# --- 4. MAIN ---
def main():
    print("=== ГЛУБОКИЙ АНАЛИЗ (Сравнение Old vs New V6) ===\n")

    # Загрузка данных
    if not os.path.exists(INPUT_FILE):
        print("❌ Файл CSV не найден.")
        return
    print("📂 Читаем данные...")
    # Адаптируйте usecols под ваш CSV, если структура отличается
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                         usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
        df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    except:
        # Fallback для 3-х признаков (как в скрипте V6)
        print("⚠️ Формат 5 колонок не подошел, пробуем формат V6 (1, 3, 5)...")
        df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                         usecols={1: 'feature_1', 3: 'feature_3', 5: 'target'}.keys())
        df.columns = ['feature_1', 'feature_3', 'target']

    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)
    X = df.drop(columns=['target'])
    y_real = df['target'].values

    # Загрузка моделей
    print("🔄 Загрузка моделей...")
    try:
        # Пытаемся загрузить, если файлы существуют
        if os.path.exists(OLD_MODEL_PATH):
            old_wrapper = joblib.load(OLD_MODEL_PATH)
            print("   ✅ Старая модель загружена.")
        else:
            old_wrapper = None
            print("   ⚠️ Старая модель не найдена, пропускаем.")

        new_wrapper = joblib.load(NEW_MODEL_PATH)
        print("   ✅ Новая модель (V6) загружена.")
    except Exception as e:
        print(f"❌ Ошибка загрузки моделей: {e}")
        return

    # --- СБОР И АНАЛИЗ ---
    all_corrected_preds = []

    # Настройка графиков
    # Рассчитаем сколько всего моделей (3 старых + 3 новых) = 6
    n_rows = 3
    n_cols = 4  # (hist, scatter) * 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 15))
    axes_flat = axes.flatten()

    # Сетка графиков: пары (hist, scatter)
    model_axes = [(axes_flat[i * 2], axes_flat[i * 2 + 1]) for i in range(n_rows * 2)]

    current_ax_idx = 0

    # 1. СТАРАЯ МОДЕЛЬ (если есть)
    if old_wrapper is not None:
        print("\n⚙️ Обработка Old Models...")
        try:
            X_old = old_wrapper.preprocessor.transform(X)
            # Если это Stacking внутри ProductionModel/HybridModelGod
            estimators_list = old_wrapper.model.estimators_ if hasattr(old_wrapper.model, 'estimators_') else []
            names_list = old_wrapper.estimators if hasattr(old_wrapper, 'estimators') else []

            for i, (name_tuple, trained_model) in enumerate(zip(names_list, estimators_list)):
                name = name_tuple[0] if isinstance(name_tuple, tuple) else name_tuple
                print(f"   -> Processing OLD_{name}...")
                analyze_submodel(f"OLD_{name}", trained_model, X_old, y_real, model_axes[current_ax_idx][0],
                                 model_axes[current_ax_idx][1])
                current_ax_idx += 1
        except Exception as e:
            print(f"   ⚠️ Не удалось разобрать старую модель: {e}")

    # 2. НОВАЯ МОДЕЛЬ (FinalModel V6)
    print("\n⚙️ Обработка New Model (V6)...")

    # Для FinalModel нам нужно самим сделать препроцессинг, чтобы достать фичи для базовых моделей
    # Используем внутренние методы FinalModel
    X_log = new_wrapper._log_transform(X)
    X_new_transformed = new_wrapper._add_cluster_features(X_log, is_fit=False)

    # В FinalModel используется StackingRegressor, модели лежат в new_wrapper.stacking_model.estimators_
    # Имена лежат в new_wrapper.stacking_model.named_estimators_

    for name, trained_model in new_wrapper.stacking_model.named_estimators_.items():
        print(f"   -> Processing NEW_{name}...")
        ax_h, ax_s = model_axes[current_ax_idx]

        # Предсказываем и анализируем
        p_raw, p_corr = analyze_submodel(f"NEW_{name}", trained_model, X_new_transformed, y_real, ax_h, ax_s)

        all_corrected_preds.append(p_corr)
        current_ax_idx += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # --- ФИНАЛЬНЫЙ ГРАФИК НОВОГО АНСАМБЛЯ ---
    print("\n🎨 Рисуем ИТОГОВЫЙ ГРАФИК (Новый Ансамбль)...")

    if all_corrected_preds:
        ensemble_corrected = np.mean(all_corrected_preds, axis=0)
        bias_corrected = y_real - ensemble_corrected
        rmse_final = np.sqrt(mean_squared_error(y_real, ensemble_corrected))

        plt.figure(figsize=(12, 6))
        sample_idx = np.random.choice(len(bias_corrected), size=min(100000, len(bias_corrected)), replace=False)

        sns.histplot(bias_corrected[sample_idx], bins=100, color='green', alpha=0.6, label=f'Corrected Ensemble V6',
                     kde=True)
        plt.axvline(0, color='red', linestyle='--', lw=2)
        plt.title(f"ФИНАЛ V6: Распределение Байаса (исправленные базовые модели)\nRMSE: {rmse_final:.5f}")
        plt.xlabel("Error (Real - Pred)")
        plt.legend()
        plt.show()

        print(f"\n✅ ИТОГОВЫЙ БАЙАС: {bias_corrected.mean():.7f}")
        print(f"✅ ИТОГОВЫЙ RMSE: {rmse_final:.5f}")
    else:
        print("⚠️ Нет данных для построения финального графика.")


if __name__ == "__main__":
    main()