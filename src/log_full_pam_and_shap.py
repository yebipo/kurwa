# =========================================================================================
# === СКРИПТ V24: BIG DATA + SHAP PNG + RESIDUAL ANALYSIS PLOT ===
# === (Добавлен график анализа погрешностей как на скриншоте) ===
# =========================================================================================
import pandas as pd
import numpy as np
import sqlite3
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings
import optuna
import matplotlib.pyplot as plt
import logging
import sys
import os
import shap
from datetime import datetime
from yellowbrick.cluster import KElbowVisualizer

# --- НАСТРОЙКИ ---
LOG_FILENAME = 'pipeline_v24_residuals.log'
logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
file_handler = logging.FileHandler(LOG_FILENAME, encoding='utf-8')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.style.use('seaborn-v0_8-whitegrid')

# КОНФИГУРАЦИЯ
FILE_PATH = 'output_adequate.csv'
RANDOM_STATE = 42
N_TRIALS_OPTUNA = 70
OPTUNA_SUB_SAMPLE = 50000
SHAP_SAMPLE_SIZE = 100
MAX_CLUSTERS_TO_TEST = 7
TEST_SIZE_SHAP = 0.20

HISTORY_DB_NAME = 'pipeline_history.db'
OPTUNA_STORAGE = "sqlite:///optuna_v24.db"
OPTUNA_STUDY_NAME = "stacking_v24"
SHAP_PLOTS_DIR = 'shap_plots_v24_final'

COLUMN_MAPPING = {1: 'feature_1', 2: 'feature_2', 3: 'feature_3', 4: 'feature_4', 5: 'target'}
FEATURE_COLUMNS = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
TARGET_COLUMN = 'target'


# --- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---
def init_history_db():
    try:
        conn = sqlite3.connect(HISTORY_DB_NAME)
        c = conn.cursor()
        c.execute(
            '''CREATE TABLE IF NOT EXISTS execution_log (id INTEGER PRIMARY KEY, timestamp TEXT, stage TEXT, status TEXT, details TEXT)''')
        conn.commit()
        conn.close()
    except:
        pass


def log_stage_to_db(stage, status, details=None):
    try:
        conn = sqlite3.connect(HISTORY_DB_NAME)
        c = conn.cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        d_json = json.dumps(details, ensure_ascii=False) if details else ""
        c.execute("INSERT INTO execution_log (timestamp, stage, status, details) VALUES (?, ?, ?, ?)",
                  (ts, stage, status, d_json))
        conn.commit()
        conn.close()
        logger.info(f"💾 [DB] {stage}: {status}")
    except:
        pass


def find_optimal_clusters(X_raw, max_clusters):
    if len(X_raw) > 100000:
        X_sample = X_raw.sample(50000, random_state=RANDOM_STATE)
    else:
        X_sample = X_raw
    try:
        X_log = np.log2(X_sample.astype(float).clip(lower=1))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_log)
        model = KMeans(random_state=RANDOM_STATE, n_init='auto')
        visualizer = KElbowVisualizer(model, k=(2, max_clusters), metrics='inertia', timings=False)
        visualizer.fit(X_scaled)
        k = visualizer.elbow_value_
        plt.close()
        return k
    except:
        return 3


# --- КЛАСС ПРЕПРОЦЕССОРА ---
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


# --- OPTUNA ---
def objective(trial, X_processed, y_target):
    X_train, X_val, y_train, y_val = train_test_split(X_processed, y_target, test_size=0.25, random_state=RANDOM_STATE)
    lgbm_p = {
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 120),
        'max_depth': trial.suggest_int('lgbm_max_depth', 5, 15),
        'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 10, 100),
        'lambda_l1': trial.suggest_float('lgbm_lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lgbm_lambda_l2', 1e-8, 10.0, log=True),
        'random_state': RANDOM_STATE, 'n_jobs': 1, 'verbose': -1
    }
    xgb_p = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 200, 1000),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
        'subsample': 0.7, 'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE, 'n_jobs': 1
    }
    cat_p = {
        'iterations': trial.suggest_int('cat_iterations', 200, 1000),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('cat_depth', 4, 8),
        'random_state': RANDOM_STATE, 'verbose': 0
    }
    est = [('l', lgb.LGBMRegressor(**lgbm_p)), ('x', xgb.XGBRegressor(**xgb_p)), ('c', cb.CatBoostRegressor(**cat_p))]
    model = StackingRegressor(estimators=est, final_estimator=LinearRegression(), cv=3, n_jobs=1)
    model.fit(X_train, y_train)
    return np.sqrt(mean_squared_error(y_val, model.predict(X_val)))


def progress_callback(study, trial):
    n = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if n % 5 == 0 or n == N_TRIALS_OPTUNA:
        logger.info(f"  >> Trial {n}/{N_TRIALS_OPTUNA}. RMSE: {trial.value:.4f}. Best: {study.best_value:.4f}")


# --- ФИНАЛЬНАЯ МОДЕЛЬ ---
class ProductionModel:
    def __init__(self, preprocessor, lgbm_p, xgb_p, cat_p):
        self.preprocessor = preprocessor
        self.estimators = [('lgbm', lgb.LGBMRegressor(**lgbm_p)), ('xgb', xgb.XGBRegressor(**xgb_p)),
                           ('catboost', cb.CatBoostRegressor(**cat_p))]
        self.model = StackingRegressor(estimators=self.estimators, final_estimator=LinearRegression(), cv=3, n_jobs=1)

    def fit(self, X_raw, y):
        X_ready = self.preprocessor.transform(X_raw)
        self.model.fit(X_ready, y)
        return self

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        return self.model.predict(X_ready)


# ========================================================
# === НОВАЯ ФУНКЦИЯ ОТРИСОВКИ (КАК НА СКРИНШОТЕ) ===
# ========================================================
def plot_residuals_analysis(y_true, y_pred, save_path):
    """Строит график Предсказание vs Реальность и график остатков."""
    # Если данных слишком много, берем сэмпл для графика, чтобы не висло
    if len(y_true) > 10000:
        indices = np.random.choice(len(y_true), 10000, replace=False)
        y_true_plot = y_true.iloc[indices] if isinstance(y_true, pd.Series) else y_true[indices]
        y_pred_plot = y_pred[indices]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred

    residuals = y_true_plot - y_pred_plot

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle('Анализ погрешностей предсказаний модели', fontsize=16)

    # 1. График: Реальность vs Предсказание
    axes[0].scatter(y_true_plot, y_pred_plot, alpha=0.5, edgecolors='k', s=30, color='#1f77b4')
    min_val = min(np.min(y_true_plot), np.min(y_pred_plot))
    max_val = max(np.max(y_true_plot), np.max(y_pred_plot))
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')
    axes[0].set_title('Предсказания vs. Реальные значения', fontsize=14)
    axes[0].set_xlabel('Реальные значения', fontsize=12)
    axes[0].set_ylabel('Предсказанные значения', fontsize=12)
    axes[0].legend()
    axes[0].grid(True)

    # 2. График: Residual Plot (Проверка на систематическую ошибку)
    axes[1].scatter(y_pred_plot, residuals, alpha=0.5, edgecolors='k', s=30, color='#1f77b4')
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2, label='Нулевая погрешность')
    axes[1].set_title('График погрешностей (Residual Plot)', fontsize=14)
    axes[1].set_xlabel('Предсказанные значения', fontsize=12)
    axes[1].set_ylabel('Погрешность (Реальное - Предсказанное)', fontsize=12)
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Отступ под заголовок
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


# --- MAIN PIPELINE ---
def main():
    init_history_db()
    log_stage_to_db("Pipeline Start", "Started", {"version": "V24 Residuals Plot"})

    # 1. Загрузка
    logger.info("--- 1. Загрузка данных ---")
    try:
        df = pd.read_csv(FILE_PATH, decimal=',', sep=';', header=None, skiprows=1,
                         usecols=COLUMN_MAPPING.keys()).rename(columns=COLUMN_MAPPING)
        df.dropna(subset=[TARGET_COLUMN], inplace=True)
        X_full = df[FEATURE_COLUMNS]
        y_full = df[TARGET_COLUMN]
    except Exception as e:
        logger.error(f"Load Error: {e}")
        return

    X_remaining, X_test, y_remaining, y_test = train_test_split(X_full, y_full, test_size=TEST_SIZE_SHAP,
                                                                random_state=RANDOM_STATE)
    logger.info(f"  Data Split: Test={len(X_test)} | Remaining={len(X_remaining)}")

    # 2. Кластеризация
    logger.info("--- 2. Кластеризация ---")
    optimal_k = find_optimal_clusters(X_remaining, MAX_CLUSTERS_TO_TEST)
    if not optimal_k: optimal_k = 3

    scaler_global = StandardScaler()
    X_rem_log = np.log2(X_remaining.astype(float).clip(lower=1))
    X_rem_scaled = scaler_global.fit_transform(X_rem_log)
    kmeans_global = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init='auto')
    kmeans_global.fit(X_rem_scaled)
    global_preprocessor = GlobalPreprocessor(scaler_global, kmeans_global)
    log_stage_to_db("Clustering", "Success", {"k": int(optimal_k)})

    # 3. Optuna
    X_optuna_full, X_train_full, y_optuna_full, y_train_full = train_test_split(X_remaining, y_remaining, test_size=0.5,
                                                                                random_state=RANDOM_STATE)

    # Sampling
    if len(X_optuna_full) > OPTUNA_SUB_SAMPLE:
        X_optuna_small = X_optuna_full.sample(n=OPTUNA_SUB_SAMPLE, random_state=RANDOM_STATE)
        y_optuna_small = y_optuna_full.loc[X_optuna_small.index]
    else:
        X_optuna_small = X_optuna_full;
        y_optuna_small = y_optuna_full

    logger.info(f"--- 3. Optuna ({N_TRIALS_OPTUNA} trials) ---")
    X_optuna_ready = global_preprocessor.transform(X_optuna_small)
    study = optuna.create_study(direction='minimize', storage=OPTUNA_STORAGE, study_name=OPTUNA_STUDY_NAME,
                                load_if_exists=True)
    study.optimize(lambda t: objective(t, X_optuna_ready, y_optuna_small), n_trials=N_TRIALS_OPTUNA, n_jobs=-1,
                   callbacks=[progress_callback])
    bp = study.best_params

    LGBM_BP = {k.replace('lgbm_', ''): v for k, v in bp.items() if k.startswith('lgbm_')}
    XGB_BP = {k.replace('xgb_', ''): v for k, v in bp.items() if k.startswith('xgb_')}
    CAT_BP = {k.replace('cat_', ''): v for k, v in bp.items() if k.startswith('cat_')}
    LGBM_BP.update({'random_state': RANDOM_STATE, 'n_jobs': 1, 'verbose': -1})
    XGB_BP.update({'random_state': RANDOM_STATE, 'n_jobs': 1})
    CAT_BP.update({'random_state': RANDOM_STATE, 'verbose': 0})

    # 4. Обучение
    logger.info("--- 4. Финальное обучение (на Full Train) ---")
    final_model = ProductionModel(global_preprocessor, LGBM_BP, XGB_BP, CAT_BP)
    final_model.fit(X_train_full, y_train_full)
    logger.info("  ✅ Модель обучена.")

    # 5. Валидация и Графики точности
    logger.info("--- 5. Валидация и Графики точности ---")
    y_pred = final_model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    logger.info(f"  RMSE: {rmse:.4f} | R2: {r2:.4f}")
    log_stage_to_db("Evaluation", "Success", {"RMSE": rmse, "R2": r2})

    # === СОХРАНЕНИЕ ГРАФИКА RESIDUAL ANALYSIS ===
    if not os.path.exists(SHAP_PLOTS_DIR): os.makedirs(SHAP_PLOTS_DIR)
    res_plot_path = os.path.join(SHAP_PLOTS_DIR, "0_residual_analysis.png")
    plot_residuals_analysis(y_test, y_pred, res_plot_path)
    logger.info(f"  ✅ График 'Анализ погрешностей' сохранен: {res_plot_path}")

    # 6. SHAP (PNGs)
    logger.info("--- 6. Генерация SHAP (Beeswarm + Dependence) ---")
    X_bg = global_preprocessor.transform(X_train_full.sample(50, random_state=RANDOM_STATE))
    X_shap_target = global_preprocessor.transform(X_test.sample(SHAP_SAMPLE_SIZE, random_state=RANDOM_STATE))

    explainer = shap.KernelExplainer(final_model.model.predict, X_bg)
    shap_values = explainer.shap_values(X_shap_target)

    # Beeswarm
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap_target, show=False)
    plt.savefig(f"{SHAP_PLOTS_DIR}/1_beeswarm.png", bbox_inches='tight');
    plt.close()

    # Bar
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_shap_target, plot_type="bar", show=False)
    plt.savefig(f"{SHAP_PLOTS_DIR}/2_importance_bar.png", bbox_inches='tight');
    plt.close()

    # Dependence
    for col in X_shap_target.columns:
        try:
            plt.figure(figsize=(8, 5))
            shap.dependence_plot(col, shap_values, X_shap_target, show=False)
            safe_col = col.replace(" ", "_")
            plt.savefig(f"{SHAP_PLOTS_DIR}/3_dependence_{safe_col}.png", bbox_inches='tight');
            plt.close()
        except:
            pass

    logger.info(f"✅ Все готово! Графики в папке: {SHAP_PLOTS_DIR}")
    log_stage_to_db("Pipeline End", "Completed")


if __name__ == "__main__":
    main()