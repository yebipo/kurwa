import pandas as pd
import numpy as np
import joblib
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

# --- НАСТРОЙКИ ---
FILE_PATH = 'million.csv'
MODEL_FILENAME = "final_model_besk_not_quantile.joblib"  # Новое имя
RANDOM_STATE = 42
MAX_CLUSTERS_TO_TEST = 7

# (Твои параметры BEST_PARAMS оставляем без изменений)
BEST_PARAMS = {
    'cat_depth': 7,
    'cat_iterations': 1332,
    'cat_l2_leaf_reg': 0.1774161150221201,
    'cat_learning_rate': 0.02466958552456176,
    'lgbm_lambda_l1': 2.6508280629175112e-08,
    'lgbm_lambda_l2': 0.009910745904773404,
    'lgbm_learning_rate': 0.029914617989631492,
    'lgbm_max_depth': 24,
    'lgbm_min_child_samples': 14,
    'lgbm_n_estimators': 1037,
    'lgbm_num_leaves': 192,
    'xgb_colsample_bytree': 0.9394261401745961,
    'xgb_learning_rate': 0.036028372449179814,
    'xgb_max_depth': 11,
    'xgb_n_estimators': 790,
    'xgb_subsample': 0.9820423568547401
}
COLUMN_MAPPING = {1: 'feature_1', 2: 'feature_2', 3: 'feature_3', 4: 'feature_4', 5: 'target'}
FEATURE_COLUMNS = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
TARGET_COLUMN = 'target'
warnings.filterwarnings('ignore')

# --- МАТЕМАТИКА ---
EPSILON = 1e-5


def to_logit(y):
    y_safe = np.clip(y, EPSILON, 1 - EPSILON)
    return np.log(y_safe / (1 - y_safe))


def from_logit(z):
    return 1 / (1 + np.exp(-z))


# --- КЛАССЫ ---
class GlobalPreprocessor:
    def __init__(self, scaler, kmeans):
        self.scaler = scaler
        self.kmeans = kmeans

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        cluster_labels = self.kmeans.predict(X_scaled)
        X_out = X_log.copy()
        X_out['cluster_id'] = cluster_labels
        return X_out


class ProductionModelLogit:
    def __init__(self, preprocessor, lgbm_p, xgb_p, cat_p):
        self.preprocessor = preprocessor

        # Важно: меняем параметры loss функций на регрессию (RMSE),
        # так как мы будем предсказывать Logit (число от -inf до +inf), а не вероятность

        # LGBM
        lgbm_p['objective'] = 'regression'
        lgbm_p['metric'] = 'rmse'

        # XGB
        xgb_p['objective'] = 'reg:squarederror'

        # CatBoost
        cat_p['loss_function'] = 'RMSE'

        self.estimators = [
            ('lgbm', lgb.LGBMRegressor(**lgbm_p)),
            ('xgb', xgb.XGBRegressor(**xgb_p)),
            ('catboost', cb.CatBoostRegressor(**cat_p))
        ]

        # Используем Ridge вместо LinearRegression для стабильности на хвостах
        self.model = StackingRegressor(
            estimators=self.estimators,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=1
        )

    def fit(self, X_raw, y_real_probability):
        X_ready = self.preprocessor.transform(X_raw)

        # !!! МАГИЯ ЗДЕСЬ !!!
        # Мы конвертируем 0.99 -> 4.59, 0.5 -> 0.0
        y_logit = to_logit(y_real_probability)

        print(f"   Note: Training on Logits. Range: {y_logit.min():.2f} to {y_logit.max():.2f}")
        self.model.fit(X_ready, y_logit)
        return self

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        # Модель предсказывает Logit
        pred_logit = self.model.predict(X_ready)
        # Конвертируем обратно в вероятность
        return from_logit(pred_logit)


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def find_optimal_clusters_math(X_raw, max_k):
    # (Оставляем твою функцию поиска кластеров без изменений)
    if len(X_raw) > 50000:
        X_sample = X_raw.sample(50000, random_state=RANDOM_STATE)
    else:
        X_sample = X_raw
    X_log = np.log2(X_sample.astype(float).clip(lower=1))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    inertias = []
    K_range = range(2, max_k + 1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        km.fit(X_scaled)
        inertias.append(km.inertia_)
    x_points = np.array(list(K_range));
    y_points = np.array(inertias)
    p1 = np.array([x_points[0], y_points[0]]);
    p2 = np.array([x_points[-1], y_points[-1]])
    distances = []
    for i in range(len(x_points)):
        p0 = np.array([x_points[i], y_points[i]])
        num = np.abs(np.cross(p2 - p1, p1 - p0))
        denom = np.linalg.norm(p2 - p1);
        distances.append(num / denom)
    best_k = x_points[np.argmax(distances)]
    print(f"  > Optimal k = {best_k}")
    return best_k


# --- MAIN ---
def main():
    print(f"=== TRAINING: LOGIT-SPACE STACKING ===")

    if not os.path.exists(FILE_PATH): return
    print("--- 1. Loading Data ---")
    df = pd.read_csv(FILE_PATH, decimal=',', sep=';', header=None, skiprows=1, usecols=COLUMN_MAPPING.keys()).rename(
        columns=COLUMN_MAPPING)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    X_full = df[FEATURE_COLUMNS]
    y_full = df[TARGET_COLUMN].values

    # Preprocessing
    k = find_optimal_clusters_math(X_full, MAX_CLUSTERS_TO_TEST)
    scaler_global = StandardScaler()
    X_log = np.log2(X_full.astype(float).clip(lower=1))
    X_scaled = scaler_global.fit_transform(X_log)
    kmeans_global = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
    kmeans_global.fit(X_scaled)
    preprocessor = GlobalPreprocessor(scaler_global, kmeans_global)

    # Params
    lgbm_bp = {k.replace('lgbm_', ''): v for k, v in BEST_PARAMS.items() if k.startswith('lgbm_')}
    xgb_bp = {k.replace('xgb_', ''): v for k, v in BEST_PARAMS.items() if k.startswith('xgb_')}
    cat_bp = {k.replace('cat_', ''): v for k, v in BEST_PARAMS.items() if k.startswith('cat_')}
    lgbm_bp.update({'random_state': RANDOM_STATE, 'n_jobs': 1, 'verbose': -1})
    xgb_bp.update({'random_state': RANDOM_STATE, 'n_jobs': 1})
    cat_bp.update({'random_state': RANDOM_STATE, 'verbose': 0, 'allow_writing_files': False})

    print("--- 2. Training Logit Model... ---")
    # Создаем новую Logit модель
    model = ProductionModelLogit(preprocessor, lgbm_bp, xgb_bp, cat_bp)

    # В fit передаем ОБЫЧНЫЕ вероятности, конвертация произойдет внутри
    model.fit(X_full, y_full)

    print(f"--- 3. Saving to {MODEL_FILENAME} ---")
    joblib.dump(model, MODEL_FILENAME)
    print("Done. Try checking this model.")


if __name__ == "__main__":
    main()