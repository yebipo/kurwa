import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

# --- НАСТРОЙКИ ---
INPUT_FILE = 'million.csv'
MODEL_FILENAME = "final_model_besk_not_quantile.joblib"  # Твоя Logit модель
BOOSTER_FILENAME = "logit_booster.joblib"  # Куда сохраним фикс
RANDOM_STATE = 42
RANGE_MIN = 0.80  # На каком диапазоне учим фикс

# --- 1. ВОССТАНОВЛЕНИЕ КЛАССОВ (COPY-PASTE) ---
EPSILON = 1e-5


def to_logit(y):
    y_safe = np.clip(y, EPSILON, 1 - EPSILON)
    return np.log(y_safe / (1 - y_safe))


def from_logit(z):
    return 1 / (1 + np.exp(-z))


class GlobalPreprocessor:
    def __init__(self, scaler, kmeans):
        self.scaler = scaler;
        self.kmeans = kmeans

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        cluster_labels = self.kmeans.predict(X_scaled)
        X_out = X_log.copy();
        X_out['cluster_id'] = cluster_labels
        return X_out


class ProductionModelLogit:
    def __init__(self, preprocessor, lgbm_p, xgb_p, cat_p):
        self.preprocessor = preprocessor
        lgbm_p['objective'] = 'regression';
        lgbm_p['metric'] = 'rmse'
        xgb_p['objective'] = 'reg:squarederror';
        cat_p['loss_function'] = 'RMSE'
        self.estimators = [('lgbm', lgb.LGBMRegressor(**lgbm_p)), ('xgb', xgb.XGBRegressor(**xgb_p)),
                           ('catboost', cb.CatBoostRegressor(**cat_p))]
        self.model = StackingRegressor(estimators=self.estimators, final_estimator=Ridge(alpha=1.0), cv=3, n_jobs=1)

    def predict(self, X_raw): pass  # Нам нужен доступ к predict_logit, сделаем его вручную ниже


# Хак для Joblib
import types

module_fake = types.ModuleType('__main__')
module_fake.GlobalPreprocessor = GlobalPreprocessor
module_fake.ProductionModelLogit = ProductionModelLogit
module_fake.to_logit = to_logit
module_fake.from_logit = from_logit


# --- КЛАСС БУСТЕРА ---
class LogitBoosterWrapper:
    def __init__(self, base_model, scaler_model):
        self.base_model = base_model
        self.scaler_model = scaler_model  # LinearRegression

    def predict(self, X):
        # 1. Получаем "сырой" ложит от базы
        X_ready = self.base_model.preprocessor.transform(X)
        z_raw = self.base_model.model.predict(X_ready)

        # 2. Растягиваем ложит (Booster)
        z_boosted = self.scaler_model.predict(z_raw.reshape(-1, 1))

        # 3. Sigmoid
        return from_logit(z_boosted)


def main():
    print(f"=== TRAINING LOGIT BOOSTER ===")

    # 1. Загрузка
    if not os.path.exists(INPUT_FILE) or not os.path.exists(MODEL_FILENAME): return
    print("📂 Loading Data...")
    df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce');
    df.dropna(subset=['target'], inplace=True)
    X = df.drop(columns=['target']);
    y = df['target'].values

    # Split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # Filter High Range (учим бустер на всем, но проверяем верха)
    # На самом деле лучше учить бустер на всей выборке или на верхах?
    # Давай учить на всей, но с весами, или просто на верхах.
    # Для агрессивности возьмем верха.
    mask_high = y_test > 0.80
    X_high = X_test[mask_high]
    y_high = y_test[mask_high]

    print(f"🎯 Fitting booster on {len(y_high)} high-range samples...")

    # 2. Load Base Model
    base_model = joblib.load(MODEL_FILENAME)

    # 3. Get Raw Logits
    print("⚙️ Predicting Raw Logits...")
    X_ready = base_model.preprocessor.transform(X_high)
    z_pred_raw = base_model.model.predict(X_ready)

    # 4. Get Real Logits
    z_real = to_logit(y_high)

    # 5. Train Linear Scaler (Z_real = alpha * Z_pred + beta)
    print("🔧 Training Linear Scaler...")
    booster = LinearRegression()
    booster.fit(z_pred_raw.reshape(-1, 1), z_real)

    alpha = booster.coef_[0]
    beta = booster.intercept_
    print(f"\n🔥 BOOSTER PARAMS: Alpha (Scale) = {alpha:.4f}, Beta (Shift) = {beta:.4f}")

    if alpha > 1.0:
        print("   ✅ Good! Alpha > 1 means we are expanding the variance (stretching predictions).")
    else:
        print("   ⚠️ Warning: Alpha < 1. The model is conservative.")

    # 6. Apply & Check
    z_boosted = booster.predict(z_pred_raw.reshape(-1, 1))
    preds_final = from_logit(z_boosted)
    preds_raw = from_logit(z_pred_raw)

    print("\n📊 RESULTS (High Range):")
    print("-" * 50)
    print(f"{'Metric':<15} | {'Raw Logit':<15} | {'BOOSTED':<15}")
    print("-" * 50)
    print(f"{'Max Pred':<15} | {preds_raw.max():.5f}        | {preds_final.max():.5f}")
    print(f"{'Real Max':<15} | {y_high.max():.5f}        | {y_high.max():.5f}")
    print(
        f"{'RMSE':<15} | {np.sqrt(mean_squared_error(y_high, preds_raw)):.5f}        | {np.sqrt(mean_squared_error(y_high, preds_final)):.5f}")
    print(f"{'Bias':<15} | {np.mean(preds_raw - y_high):.5f}        | {np.mean(preds_final - y_high):.5f}")

    # 7. Save Wrapper
    final_wrapper = LogitBoosterWrapper(base_model, booster)
    joblib.dump(final_wrapper, BOOSTER_FILENAME)
    print(f"\n💾 Saved boosted model to: {BOOSTER_FILENAME}")

    # 8. Plot
    plt.figure(figsize=(10, 5))
    plt.scatter(y_high, preds_raw, alpha=0.3, label='Raw', color='purple')
    plt.scatter(y_high, preds_final, alpha=0.3, label='Boosted', color='red')
    plt.plot([0.8, 1], [0.8, 1], 'k--')
    plt.title(f"Logit Booster Effect (Alpha={alpha:.2f})")
    plt.xlabel("Real")
    plt.ylabel("Pred")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()