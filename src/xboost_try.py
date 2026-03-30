import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import os
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- CONFIG ---
INPUT_FILE = 'million.csv'
OUTPUT_FILE = "super_sniper_optuna.joblib"
N_CLUSTERS = 4  # Optimized value
RANDOM_STATE = 42
HIGH_RANGE_WEIGHT = 20.0  # Aggressive weight for > 0.75
OPTUNA_TRIALS = 50  # Enough to find good params
OPTUNA_SAMPLE = 100000  # Sample size for speed


# --- PREPROCESSOR ---
class SmartPreprocessor(BaseEstimator):
    def __init__(self, n_clusters=4):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.n_clusters = n_clusters

    def fit(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        self.scaler.fit(X_log)
        X_scaled = self.scaler.transform(X_log)
        self.kmeans.fit(X_scaled)
        return self

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

        # Math Features
        eps = 1e-4
        df['f1_f2_sum'] = df['feature_1'] + df['feature_2']
        df['f1_f2_diff'] = df['feature_1'] - df['feature_2']
        df['f3_f4_sum'] = df['feature_3'] + df['feature_4']
        df['group1_x_group2'] = df['f1_f2_sum'] * df['f3_f4_sum']
        df['f1_over_f3'] = df['feature_1'] / (df['feature_3'] + eps)
        df['f2_over_f4'] = df['feature_2'] / (df['feature_4'] + eps)
        df['row_median'] = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']].median(axis=1)

        return df


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


# --- OPTUNA OBJECTIVE ---
def objective(trial, X, y, w):
    param = {
        'n_estimators': trial.suggest_int('n_estimators', 1000, 3000),
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),

        # Hardware
        'n_jobs': -1,
        'tree_method': 'hist',
        'objective': 'reg:squarederror',
        'verbosity': 0
    }

    # Internal Split
    X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(X, y, w, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(**param)
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_va, y_va)], verbose=False)

    preds = model.predict(X_va)

    # Custom Metric: Weighted RMSE focus on High Range
    mask_h = y_va >= 0.8
    if mask_h.sum() > 0:
        rmse_h = np.sqrt(mean_squared_error(y_va[mask_h], preds[mask_h]))
        return rmse_h  # Optimize ONLY for High Range
    return np.sqrt(mean_squared_error(y_va, preds))


# --- MAIN ---
def main():
    print(f"=== SUPER SNIPER: OPTUNA TUNING ({N_CLUSTERS} clusters) ===\n")

    # 1. Load Data
    print("📂 Loading Data...")
    df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys(), engine='c')
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)

    X = df.drop(columns=['target'])
    y = df['target']

    # 2. Main Split
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_STATE)

    # 3. Weights
    print(f"⚖️ Applying Weights (x{HIGH_RANGE_WEIGHT} for > 0.75)...")
    w_train_full = np.ones(len(y_train_full))
    w_train_full[y_train_full > 0.75] = HIGH_RANGE_WEIGHT

    # 4. Fit Preprocessor
    print("⚙️ Preprocessing...")
    prep = SmartPreprocessor(n_clusters=N_CLUSTERS)
    prep.fit(X_train_full)
    X_train_trans = prep.transform(X_train_full)
    X_test_trans = prep.transform(X_test)

    # 5. Optuna
    print(f"🔍 Starting Optuna ({OPTUNA_TRIALS} trials) on sample...")
    # Sample for speed
    idx = np.random.choice(len(X_train_trans), size=min(len(X_train_trans), OPTUNA_SAMPLE), replace=False)
    X_opt = X_train_trans.iloc[idx]
    y_opt = y_train_full.iloc[idx]
    w_opt = w_train_full[idx]

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, X_opt, y_opt, w_opt), n_trials=OPTUNA_TRIALS, n_jobs=1)

    best_params = study.best_params
    best_params.update({'n_jobs': -1, 'tree_method': 'hist', 'objective': 'reg:squarederror'})
    print(f"\n✅ Best Params: {best_params}")

    # 6. Final Train
    print("\n🚀 Training FINAL Super Sniper...")
    # Increase estimators slightly for full data
    best_params['n_estimators'] = int(best_params['n_estimators'] * 1.2)

    final_model = xgb.XGBRegressor(**best_params)
    final_model.fit(
        X_train_trans, y_train_full,
        sample_weight=w_train_full,
        eval_set=[(X_test_trans, y_test)],
        verbose=100
    )

    # 7. Evaluate
    print("\n🎯 Final Evaluation...")
    wrapper = ChampionWrapper(final_model, prep)
    y_pred = wrapper.predict(X_test)

    rmse_glob = np.sqrt(mean_squared_error(y_test, y_pred))
    mask_h = (y_test >= 0.8)
    rmse_h = np.sqrt(mean_squared_error(y_test[mask_h], y_pred[mask_h])) if mask_h.sum() > 0 else 0

    print("-" * 40)
    print(f"Global RMSE:     {rmse_glob:.6f}")
    print(f"High RMSE (0.8+): {rmse_h:.6f}")
    print("-" * 40)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(y_test - y_pred, bins=100, kde=True)
    plt.title("Residuals")

    plt.subplot(1, 2, 2)
    plt.hexbin(y_test, y_pred, gridsize=40, cmap='inferno', mincnt=1, bins='log')
    plt.plot([0, 1], [0, 1], 'w--')
    plt.title(f"Super Sniper (High RMSE: {rmse_h:.5f})")
    plt.show()

    joblib.dump(wrapper, OUTPUT_FILE)
    print(f"💾 Model saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()