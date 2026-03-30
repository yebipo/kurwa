import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import os
import torch
from sklearn.base import BaseEstimator
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
INPUT_FILE = 'million.csv'
BASE_MODEL_FILE = "champion_model_final.joblib"
OUTPUT_FILE = "champion_corrected.joblib"
EPSILON = 1e-4

# --- 1. PREPROCESSOR (REQUIRED) ---
class SmartPreprocessor(BaseEstimator):
    def __init__(self, n_clusters=15):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.n_clusters = n_clusters

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)

        df['row_mean'] = df.mean(axis=1)
        df['row_std'] = df.std(axis=1)
        dists = self.kmeans.transform(X_scaled)
        for i in range(self.n_clusters):
            df[f'dist_clust_{i}'] = dists[:, i]
        df['cluster_id'] = self.kmeans.predict(X_scaled)

        df['f1_f2_sum'] = df['feature_1'] + df['feature_2']
        df['f1_f2_diff'] = df['feature_1'] - df['feature_2']
        df['f3_f4_sum'] = df['feature_3'] + df['feature_4']
        df['group1_x_group2'] = df['f1_f2_sum'] * df['f3_f4_sum']
        df['f1_over_f3'] = df['feature_1'] / (df['feature_3'] + EPSILON)
        df['f2_over_f4'] = df['feature_2'] / (df['feature_4'] + EPSILON)
        df['row_median'] = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']].median(axis=1)
        return df

# --- 2. OLD WRAPPER ---
class ChampionWrapper:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def predict(self, X_raw):
        X_trans = self.preprocessor.transform(X_raw)
        preds = self.model.predict(X_trans)
        if preds.min() < -0.5 or preds.max() > 1.5:
            preds = 1 / (1 + np.exp(-preds))
        return np.clip(preds, 0, 1)

# --- 3. NEW CORRECTED WRAPPER ---
class CorrectedChampionWrapper:
    def __init__(self, base_model, corrector_model):
        self.base_model = base_model
        self.corrector_model = corrector_model

    def predict(self, X_raw):
        # 1. Базовый прогноз
        base_preds = self.base_model.predict(X_raw)

        # 2. Предсказываем ошибку (Input: base_preds reshaped 2D)
        # Корректор думает: "Если прогноз 0.9, то ошибка обычно +0.01"
        correction = self.corrector_model.predict(base_preds.reshape(-1, 1))

        # 3. Исправляем: Prediction + Estimated_Error
        final_preds = base_preds + correction

        return np.clip(final_preds, 0, 1)

# --- 4. MAIN ---
def main():
    print("=== TRAINING RESIDUAL CORRECTOR (The Doctor) ===\n")

    # 1. Load Data
    print("📂 Loading Data...")
    if not os.path.exists(BASE_MODEL_FILE):
        print(f"❌ Файл {BASE_MODEL_FILE} не найден!")
        return

    df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys(), engine='c')
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)
    y_real = df['target'].values
    X_raw = df.drop(columns=['target'])

    # 2. Load Base Model
    print(f"🔄 Loading Champion: {BASE_MODEL_FILE}...")
    original_torch_load = torch.load
    torch.load = lambda *args, **kwargs: original_torch_load(*args, **kwargs, map_location='cpu')
    try:
        base_model = joblib.load(BASE_MODEL_FILE)
    finally:
        torch.load = original_torch_load

    # 3. Generate Predictions (Features for Corrector)
    print("🚀 Generating base predictions...")
    base_preds = base_model.predict(X_raw)

    # 4. Calculate Residuals (Target for Corrector)
    # Residual = True - Pred
    residuals = y_real - base_preds

    print(f"   Global Bias before fix: {np.mean(residuals):.6f}")

    # 5. Train Corrector (LightGBM)
    # Input: Base Prediction (1 Feature)
    # Output: Residual
    print("🎓 Training The Doctor (LightGBM on Residuals)...")

    X_corr = base_preds.reshape(-1, 1) # Feature
    y_corr = residuals                 # Target

    # Split for validation to avoid overfitting the noise
    X_tr, X_val, y_tr, y_val = train_test_split(X_corr, y_corr, test_size=0.2, random_state=42)

    # Ограничиваем глубину, чтобы он учил только явные паттерны, а не шум
    corrector = lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,      # Неглубокие деревья
        num_leaves=15,    # Мало листьев = обобщение
        n_jobs=-1,
        verbose=-1
    )

    corrector.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])

    # 6. Evaluate
    print("\n🩺 Evaluating Fix...")
    correction_val = corrector.predict(X_val)

    # Apply fix
    # Original prediction on validation set
    preds_val_orig = X_val.flatten()
    # Corrected
    preds_val_fixed = preds_val_orig + correction_val
    preds_val_fixed = np.clip(preds_val_fixed, 0, 1)

    # Real values for validation set
    # y_val is residual (real - pred), so real = pred + residual
    real_val = preds_val_orig + y_val

    def get_metrics(y_t, y_p, name):
        rmse = np.sqrt(mean_squared_error(y_t, y_p))
        mask = (y_t >= 0.8)
        rmse_h = np.sqrt(mean_squared_error(y_t[mask], y_p[mask])) if mask.sum() > 0 else 0
        print(f"{name:<15} | Global: {rmse:.6f} | High: {rmse_h:.6f}")
        return rmse, rmse_h

    print("-" * 50)
    print(f"{'TYPE':<15} | {'GLOBAL RMSE':<15} | {'HIGH RMSE'}")
    print("-" * 50)
    get_metrics(real_val, preds_val_orig, "ORIGINAL")
    get_metrics(real_val, preds_val_fixed, "CORRECTED")
    print("-" * 50)

    # 7. Visualization of the Correction Function
    # Let's see what the doctor learned
    x_range = np.linspace(0, 1, 500).reshape(-1, 1)
    y_correction = corrector.predict(x_range)

    plt.figure(figsize=(10, 6))
    plt.plot(x_range, np.zeros_like(x_range), 'k--', label='Zero Correction')
    plt.plot(x_range, y_correction, 'r-', linewidth=3, label='Learned Correction')
    plt.title("What the 'Doctor' learned:\nInput (Pred) -> Output (Add to Pred)")
    plt.xlabel("Original Model Prediction")
    plt.ylabel("Value to Add")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 8. Save
    final_wrapper = CorrectedChampionWrapper(base_model, corrector)
    joblib.dump(final_wrapper, OUTPUT_FILE)
    print(f"\n💾 Saved Corrected Model to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()