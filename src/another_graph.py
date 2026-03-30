import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import warnings
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.cluster import KMeans
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

# --- 1. НАСТРОЙКИ ---
WORK_DIR = "."
INPUT_FILE = 'million.csv'
MODEL_FILE = "final_model_unbiased.joblib"
GLOBAL_CORRECTOR_PATH = "simple_shift_corrector.joblib"

RANDOM_STATE = 42
EPSILON = 1e-5


# --- 2. КЛАССЫ (НУЖНЫ ДЛЯ ЗАГРУЗКИ) ---
def to_logit(y): return np.log(np.clip(y, EPSILON, 1 - EPSILON) / (1 - np.clip(y, EPSILON, 1 - EPSILON)))


def from_logit(z): return 1 / (1 + np.exp(-z))


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


class SmartPreprocessor:
    def __init__(self, n_clusters=20):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE);
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')

    def transform(self, X_raw): return X_raw


class IsotonicCorrector:
    def __init__(self): self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    def predict(self, y): return self.iso.predict(y)


class HybridModelGod:
    def __init__(self, p, lq, lm, xq, xm, cq, cm): pass


class CalibratedProductionModel:
    def __init__(self, base_model, calibrator): self.base_model = base_model; self.calibrator = calibrator

    def predict(self, X_raw): return self.calibrator.predict(self.base_model.predict(X_raw).reshape(-1, 1))


class AdditiveCalibratedModel:
    def __init__(self, base_model, bias_value): self.base_model = base_model; self.bias_value = float(bias_value)

    def predict(self, X_raw): return self.base_model.predict(X_raw) + self.bias_value


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
    def __init__(self, lgbm_params, xgb_params, catboost_params, n_clusters, random_state=42): pass

    def predict(self, X_raw): return np.zeros(len(X_raw))


# --- 3. BATCH PREDICT ---
def predict_batch(model, X_df):
    preds = []
    batch_size = 50000
    try:
        model.predict(X_df.iloc[:2], apply_calibration=True);
        use_calib = True
    except:
        use_calib = False
    print(f"   -> Predicting...", end=" ")
    for i in range(0, len(X_df), batch_size):
        if i % 200000 == 0: print(".", end="", flush=True)
        batch = X_df.iloc[i:i + batch_size]
        try:
            p = model.predict(batch, apply_calibration=True) if use_calib else model.predict(batch)
            if p.ndim > 1: p = p.flatten()
            preds.append(p)
        except:
            return None
    print(" Done.")
    return np.hstack(preds)


# --- 4. ФУНКЦИЯ ДЛЯ ОПТИМИЗАТОРА ---
def calc_rmse_on_interval(shift_val, y_p_subset, y_r_subset):
    """
    Считает RMSE только на переданном подмножестве (0-0.8),
    применяя простой сдвиг.
    """
    y_corr = y_p_subset + shift_val
    # clip важен, чтобы оптимизатор не штрафовал за выход за 0,
    # хотя при хорошем сдвиге этого и так не будет
    y_corr = np.clip(y_corr, 0, 1)
    return np.sqrt(mean_squared_error(y_r_subset, y_corr))


# --- 5. MAIN ---
def main():
    print(f"\n{'=' * 65}")
    print(f"=== INTERVAL-SPECIFIC OPTIMIZATION (0-0.8 ONLY) ===")
    print(f"{'=' * 65}\n")

    if not os.path.exists(INPUT_FILE): return

    # 1. Читаем данные
    print("📂 1. Читаем данные...")
    df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)
    X = df.drop(columns=['target'])
    y_real = df['target'].values

    # 2. Модель
    print(f"\n🔄 2. Загрузка модели...")
    try:
        model = joblib.load(MODEL_FILE)
    except Exception as e:
        print(f"❌ {e}"); return
    y_pred = predict_batch(model, X)
    if y_pred is None: return

    # 3. ПОДГОТОВКА ДАННЫХ ДЛЯ ОПТИМИЗАТОРА
    print("\n🎯 3. Выделяем интервал 0 <= Pred <= 0.8...")

    # Маска для оптимизации: берем только те строки, где предсказание <= 0.8
    # Именно на них мы будем учить "shift"
    OPTIM_LIMIT = 0.80
    mask_optim = y_pred <= OPTIM_LIMIT

    y_pred_subset = y_pred[mask_optim]
    y_real_subset = y_real[mask_optim]

    print(f"   -> Обучаемся на {len(y_pred_subset)} точках (из {len(y_pred)} общих)")

    # 4. ЗАПУСК ОПТИМИЗАТОРА
    print("🚀 4. Ищем идеальный сдвиг (minimize_scalar)...")

    res = minimize_scalar(
        calc_rmse_on_interval,
        bounds=(-0.2, 0.2),  # Ищем сдвиг в пределах +/- 0.2
        args=(y_pred_subset, y_real_subset),
        method='bounded'
    )

    best_shift = res.x
    best_rmse_interval = res.fun

    print(f"   ✅ НАЙДЕН ОПТИМУМ!")
    print(f"   -> Best Shift: {best_shift:.6f}")
    print(f"   -> RMSE на интервале 0-{OPTIM_LIMIT}: {best_rmse_interval:.6f}")

    # 5. ПРИМЕНЕНИЕ КОРРЕКЦИИ (С ЗАТУХАНИЕМ)
    print("\n🔧 5. Применяем сдвиг ко всему датасету (с fade out)...")

    START_FADE = 0.80
    END_FADE = 0.95

    # Веса: 1.0 если pred < 0.8, плавно до 0.0 к 0.95
    weights = np.clip((END_FADE - y_pred) / (END_FADE - START_FADE), 0, 1)

    # Формула: Просто сдвиг * Вес
    y_corrected = y_pred + (best_shift * weights)
    y_corrected = np.clip(y_corrected, 0, 1)

    # 6. МЕТРИКИ
    # Считаем RMSE на "обучающем" интервале (0-0.8) для проверки
    rmse_low_orig = np.sqrt(mean_squared_error(y_real[mask_optim], y_pred[mask_optim]))
    rmse_low_corr = np.sqrt(mean_squared_error(y_real[mask_optim], y_corrected[mask_optim]))

    # Считаем RMSE на "высоком" интервале (>0.9)
    mask_high = y_real > 0.90
    if np.sum(mask_high) > 0:
        rmse_high_orig = np.sqrt(mean_squared_error(y_real[mask_high], y_pred[mask_high]))
        rmse_high_corr = np.sqrt(mean_squared_error(y_real[mask_high], y_corrected[mask_high]))
    else:
        rmse_high_orig, rmse_high_corr = 0, 0

    rmse_global_orig = np.sqrt(mean_squared_error(y_real, y_pred))
    rmse_global_corr = np.sqrt(mean_squared_error(y_real, y_corrected))

    print("\n" + "=" * 65)
    print(f"{'METRIC':<30} | {'ORIGINAL':<12} | {'OPTIMIZED':<12}")
    print("-" * 65)
    print(f"{'RMSE (Interval 0-0.8)':<30} | {rmse_low_orig:.6f}     | {rmse_low_corr:.6f}")
    print(f"{'RMSE (Interval >0.9)':<30} | {rmse_high_orig:.6f}     | {rmse_high_corr:.6f}")
    print(f"{'Global RMSE':<30} | {rmse_global_orig:.6f}     | {rmse_global_corr:.6f}")
    print("-" * 65)

    # Сохраняем просто число (сдвиг) в файл, чтобы потом знать, на сколько двигать
    joblib.dump(best_shift, GLOBAL_CORRECTOR_PATH)
    print(f"💾 Значение сдвига сохранено: {GLOBAL_CORRECTOR_PATH}")

    # 7. ГРАФИКИ
    plt.figure(figsize=(18, 6))
    if len(y_real) > 50000:
        idx = np.random.choice(len(y_real), 50000, replace=False)
    else:
        idx = np.arange(len(y_real))

    # График 1: Оптимизация (парабола RMSE от сдвига)
    plt.subplot(1, 3, 1)
    shifts = np.linspace(best_shift - 0.05, best_shift + 0.05, 50)
    # Считаем RMSE только для подмножества
    rmses = [calc_rmse_on_interval(s, y_pred_subset[:5000], y_real_subset[:5000]) for s in shifts]
    plt.plot(shifts, rmses, 'b-', label='Error Surface')
    plt.plot(best_shift, calc_rmse_on_interval(best_shift, y_pred_subset[:5000], y_real_subset[:5000]), 'ro', ms=10,
             label='Optimum')
    plt.title(f"Минимизация на интервале 0-0.8\nOptimum: {best_shift:.5f}")
    plt.xlabel("Величина сдвига")
    plt.ylabel("RMSE (0-0.8)")
    plt.legend()
    plt.grid(True)

    # График 2: ДО vs ПОСЛЕ (Scatter)
    plt.subplot(1, 3, 2)
    plt.scatter(y_real[idx], y_pred[idx], alpha=0.1, s=1, c='blue', label='Orig')
    plt.scatter(y_real[idx], y_corrected[idx], alpha=0.1, s=1, c='orange', label='Corr')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title("Scatter Plot")
    plt.legend()

    # График 3: Residuals
    plt.subplot(1, 3, 3)
    res = y_real[idx] - y_corrected[idx]
    plt.scatter(y_corrected[idx], res, alpha=0.1, s=1, c='green')
    plt.axhline(0, c='red', ls='--')
    plt.axvline(0.8, c='k', ls=':', label='End of Shift')
    plt.title("Residuals (Сдвиг до 0.8)")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()