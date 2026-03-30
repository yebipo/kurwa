import pandas as pd
import numpy as np
import joblib
import os
import optuna
import time
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

# --- НАСТРОЙКИ ---
FILE_PATH = 'million.csv'
MODEL_FILENAME = "final_model_entropy_fast.joblib"
DB_NAME = "sqlite:///optuna_entropy_fast.db"
STUDY_NAME = "entropy_stacking_logit_fast_v1"
RANDOM_STATE = 42
N_TRIALS_OPTUNA = 50  # Общий план итераций
OPTUNA_SAMPLE_SIZE = 150000  # Размер выборки для поиска (ускорение)
TEST_SIZE = 0.20

# Ищем 95-й перцентиль
QUANTILE_ALPHA = 0.95
EPSILON = 1e-5

COLUMN_MAPPING = {1: 'feature_1', 2: 'feature_2', 3: 'feature_3', 4: 'feature_4', 5: 'target'}
FEATURE_COLUMNS = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
TARGET_COLUMN = 'target'

warnings.filterwarnings('ignore')


# --- ТРАНСФОРМАЦИЯ ТАРГЕТА (LOGIT) ---
def to_logit(y):
    """Растягивает [0, 1] в бесконечность."""
    y_safe = np.clip(y, EPSILON, 1 - EPSILON)
    return np.log(y_safe / (1 - y_safe))


def from_logit(z):
    """Возвращает предсказания в диапазон [0, 1]"""
    return 1 / (1 + np.exp(-z))


# --- PINBALL LOSS ---
def pinball_loss(y_true, y_pred, alpha):
    residual = y_true - y_pred
    return np.mean(np.maximum(alpha * residual, (alpha - 1) * residual))


# --- ПРЕПРОЦЕССИНГ ---
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


# --- МОДЕЛЬ (STACKING) ---
class ProductionModel:
    def __init__(self, preprocessor, lgbm_p, xgb_p, cat_p):
        self.preprocessor = preprocessor

        # Настройка на все ядра (-1)
        lgbm_p.update({
            'objective': 'quantile', 'alpha': QUANTILE_ALPHA, 'metric': 'quantile',
            'verbose': -1, 'n_jobs': -1, 'linear_tree': True
        })
        xgb_p.update({
            'objective': 'reg:quantileerror', 'quantile_alpha': QUANTILE_ALPHA,
            'verbosity': 0, 'n_jobs': -1
        })
        cat_p.update({
            'loss_function': f'Quantile:alpha={QUANTILE_ALPHA}',
            'verbose': 0, 'allow_writing_files': False, 'thread_count': -1
        })

        self.estimators = [
            ('lgbm', lgb.LGBMRegressor(**lgbm_p)),
            ('xgb', xgb.XGBRegressor(**xgb_p)),
            ('catboost', cb.CatBoostRegressor(**cat_p))
        ]

        final_layer = Ridge(alpha=0.1, fit_intercept=True, random_state=RANDOM_STATE)

        self.model = StackingRegressor(
            estimators=self.estimators,
            final_estimator=final_layer,
            cv=3,
            n_jobs=1,
            passthrough=True
        )

    def fit(self, X_raw, y_transformed, weights=None):
        X_ready = self.preprocessor.transform(X_raw)
        self.model.fit(X_ready, y_transformed, sample_weight=weights)
        return self

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        return self.model.predict(X_ready)


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def find_optimal_clusters_math(X_raw, max_k):
    sample_size = min(50000, len(X_raw))
    X_sample = X_raw.sample(sample_size, random_state=RANDOM_STATE)

    X_log = np.log2(X_sample.astype(float).clip(lower=1))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    inertias = []
    K_range = range(2, max_k + 1)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    x_points = np.array(list(K_range))
    y_points = np.array(inertias)
    p1 = np.array([x_points[0], y_points[0]])
    p2 = np.array([x_points[-1], y_points[-1]])
    distances = []
    for i in range(len(x_points)):
        p0 = np.array([x_points[i], y_points[i]])
        num = np.abs(np.cross(p2 - p1, p1 - p0))
        denom = np.linalg.norm(p2 - p1)
        distances.append(num / denom)
    best_k = x_points[np.argmax(distances)]




def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mask = y_true != 0
    mape = (np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])).mean() * 100
    return rmse, mae, mape, r2


# --- OPTUNA OBJECTIVE ---
def objective(trial, X, y_transformed, preprocessor, original_weights):
    lgbm_params = {
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 500, 1500),
        'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 100),
        'max_depth': trial.suggest_int('lgbm_max_depth', 3, 15),
        'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('lgbm_lambda_l1', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lgbm_lambda_l2', 1e-8, 10.0, log=True),
        'n_jobs': -1
    }
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 500, 1200),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 12),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'n_jobs': -1
    }
    cat_params = {
        'iterations': trial.suggest_int('cat_iterations', 500, 1500),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1),
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1e-3, 10, log=True),
        'thread_count': -1
    }

    model = ProductionModel(preprocessor, lgbm_params, xgb_params, cat_params)
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    scores = []
    X_ready = preprocessor.transform(X).reset_index(drop=True)
    y_reset = y_transformed.reset_index(drop=True)
    w_reset = original_weights.reset_index(drop=True)

    for train_index, val_index in kf.split(X_ready):
        X_tr, X_val = X_ready.iloc[train_index], X_ready.iloc[val_index]
        y_tr, y_val = y_reset.iloc[train_index], y_reset.iloc[val_index]
        w_tr = w_reset.iloc[train_index]

        model.model.fit(X_tr, y_tr, sample_weight=w_tr)
        preds_logit = model.model.predict(X_val)
        scores.append(pinball_loss(y_val, preds_logit, QUANTILE_ALPHA))

    return np.mean(scores)


# --- MAIN ---
def main():
    start_time = time.time()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print("=== ЗАПУСК: ВОССТАНОВЛЕНИЕ ПОСЛЕ СБОЯ ===")

    if not os.path.exists(FILE_PATH):
        print(f"❌ Файл {FILE_PATH} не найден.")
        return

    # 1. Загрузка данных
    print("--- 1. Загрузка и подготовка данных ---")
    df = pd.read_csv(FILE_PATH, decimal=',', sep=';', header=None, skiprows=1, usecols=COLUMN_MAPPING.keys()).rename(
        columns=COLUMN_MAPPING)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Трансформации
    y_train_logit = to_logit(y_train)
    weights_train = y_train ** 5 / (y_train ** 5).mean()

    # Кластеры
    print("--- 2. Восстановление кластеров ---")
    k, scaler, kmeans = find_optimal_clusters_math(X_train, 7)
    preprocessor = GlobalPreprocessor(scaler, kmeans)

    # Подготовка сэмпла для Optuna
    if len(X_train) > OPTUNA_SAMPLE_SIZE:
        sample_idx = np.random.choice(X_train.index, size=OPTUNA_SAMPLE_SIZE, replace=False)
        X_optuna = X_train.loc[sample_idx]
        y_optuna = y_train_logit.loc[sample_idx]
        w_optuna = weights_train.loc[sample_idx]
    else:
        X_optuna, y_optuna, w_optuna = X_train, y_train_logit, weights_train

    # 2. Подключение к Study и УМНЫЙ ПЕРЕЗАПУСК
    print(f"--- 3. Проверка базы данных Optuna ({DB_NAME}) ---")

    study = optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        storage=DB_NAME,
        load_if_exists=True  # Загружает историю, если база есть
    )

    # Считаем завершенные итерации
    completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    trials_left = N_TRIALS_OPTUNA - completed_trials

    print(f"   > Уже завершено: {completed_trials}")
    print(f"   > План: {N_TRIALS_OPTUNA}")

    func = lambda trial: objective(trial, X_optuna, y_optuna, preprocessor, w_optuna)

    if trials_left > 0:
        print(f"   > Осталось выполнить: {trials_left}. Запускаем...")
        with tqdm(total=trials_left, desc="🚀 Resuming Optuna", unit="trial") as pbar:
            def tqdm_callback(study, trial):
                pbar.update(1)
                pbar.set_postfix({"Best Q-Loss": f"{study.best_value:.4f}", "Current": f"{trial.value:.4f}"})

            study.optimize(func, n_trials=trials_left, n_jobs=1, callbacks=[tqdm_callback])
    else:
        print("   ✅ План выполнен (или перевыполнен). Пропускаем поиск и берем лучшие параметры.")

    best_params = study.best_params
    print("\n✅ Лучшие параметры получены.")

    # 3. Финальное обучение
    print("--- 4. Финальное обучение на 100% данных ---")
    lgbm_bp = {k.replace('lgbm_', ''): v for k, v in best_params.items() if k.startswith('lgbm_')}
    xgb_bp = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
    cat_bp = {k.replace('cat_', ''): v for k, v in best_params.items() if k.startswith('cat_')}

    model = ProductionModel(preprocessor, lgbm_bp, xgb_bp, cat_bp)
    model.fit(X_train, y_train_logit, weights=weights_train)

    # 4. Оценка
    print("--- 5. Сохранение и отчет ---")
    y_pred_logit = model.predict(X_test)
    y_pred_final = from_logit(y_pred_logit)

    rmse, mae, mape, r2 = calculate_metrics(y_test, y_pred_final)
    joblib.dump(model, MODEL_FILENAME)

    print("\n" + "=" * 52)
    print(f"===    ИТОГОВЫЙ ОТЧЕТ (RECOVERY MODE)    ===")
    print("=" * 52 + "\n")
    print(f"  Файл модели: {MODEL_FILENAME}")
    print(f"  Макс. значение (Test): {y_test.max():.4f}")
    print(f"  Макс. значение (Pred): {y_pred_final.max():.4f}")
    print("-" * 30)
    print(f"  R²:   {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print("\n" + "=" * 52)


if __name__ == "__main__":
    main()