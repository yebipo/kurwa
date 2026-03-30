# =========================================================================================
# ===     СКРИПТ V15: НАДЕЖНЫЙ ПАЙПЛАЙН С ВОЗОБНОВЛЯЕМЫМ ХРАНИЛИЩЕМ OPTUNA         ===
# ===  (SQLite для Optuna -> оценка -> SHAP-анализ -> авто-сохранение -> отчет)      ===
# =========================================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
import warnings
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import sys
import os
import shap

# Глобальные настройки
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-whitegrid')

# Включаем логирование Optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

# --- 1. КОНФИГУРАЦИЯ ---
FILE_PATH = 'output_adequate.csv'
RANDOM_STATE = 42
N_TRIALS_OPTUNA = 70
TEST_SIZE_FOR_VALIDATION = 0.2
N_CLUSTERS = 4
SHAP_PLOTS_DIR = 'shap_plots'
RESULTS_FILE_PATH = 'model_results_summary.txt'

# --- НОВЫЕ ПАРАМЕТРЫ ДЛЯ НАДЕЖНОГО ХРАНЕНИЯ OPTUNA ---
OPTUNA_STORAGE_NAME = "sqlite:///optuna_study.db"  # Имя файла базы данных
OPTUNA_STUDY_NAME = "stacking_model_optimization_v15"  # Уникальное имя для этого исследования

COLUMN_MAPPING = {1: 'feature_1', 3: 'feature_3', 5: 'target'}
FEATURE_COLUMNS = ['feature_1', 'feature_3']
TARGET_COLUMN = 'target'


# --- 2. КЛАСС ФИНАЛЬНОЙ МОДЕЛИ (без изменений) ---
class FinalModel:
    def __init__(self, lgbm_params, xgb_params, catboost_params, n_clusters, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters;
        self.random_state = random_state;
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.feature_names_in_ = None
        estimators = [('lgbm', lgb.LGBMRegressor(**lgbm_params)), ('xgb', xgb.XGBRegressor(**xgb_params)),
                      ('catboost', cb.CatBoostRegressor(**catboost_params))]
        self.stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=3,
                                                n_jobs=-1, verbose=0)

    def _log_transform(self, X):
        X_transformed = X.copy();
        for col in X_transformed.columns: X_transformed[col] = np.log2(X_transformed[col].astype(float).clip(lower=1))
        return X_transformed

    def _add_cluster_features(self, X_log_transformed, is_fit=False):
        if is_fit:
            X_scaled = self.scaler.fit_transform(X_log_transformed); cluster_labels = self.kmeans.fit_predict(X_scaled)
        else:
            X_scaled = self.scaler.transform(X_log_transformed); cluster_labels = self.kmeans.predict(X_scaled)
        X_with_cluster = X_log_transformed.copy();
        X_with_cluster['cluster_id'] = cluster_labels
        return X_with_cluster

    def fit(self, X_raw, y):
        self.feature_names_in_ = X_raw.columns.tolist();
        X_log = self._log_transform(X_raw)
        X_final = self._add_cluster_features(X_log, is_fit=True);
        self.stacking_model.fit(X_final, y)
        return self

    def predict(self, X_raw):
        if X_raw.columns.tolist() != self.feature_names_in_: raise ValueError(f"Неверные имена колонок!")
        X_log = self._log_transform(X_raw);
        X_final = self._add_cluster_features(X_log, is_fit=False)
        return self.stacking_model.predict(X_final)

    def get_final_features(self, X_raw):
        if not hasattr(self.scaler, 'mean_'): raise RuntimeError("Модель еще не обучена.")
        X_log = self._log_transform(X_raw);
        X_final_features = self._add_cluster_features(X_log, is_fit=False)
        return X_final_features


# --- 3. ФУНКЦИЯ ДЛЯ OPTUNA (без изменений) ---
def objective(trial, X_processed, y_target):
    lgbm_params = {'n_estimators': trial.suggest_int('lgbm_n_estimators', 400, 2000),
                   'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.1, log=True),
                   'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 150),
                   'max_depth': trial.suggest_int('lgbm_max_depth', 5, 20),
                   'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 5, 100),
                   'lambda_l1': trial.suggest_float('lgbm_lambda_l1', 1e-8, 10.0, log=True),
                   'lambda_l2': trial.suggest_float('lgbm_lambda_l2', 1e-8, 10.0, log=True),
                   'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1}
    xgb_params = {'n_estimators': trial.suggest_int('xgb_n_estimators', 400, 2000),
                  'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
                  'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                  'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                  'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                  'random_state': RANDOM_STATE, 'n_jobs': -1}
    catboost_params = {'iterations': trial.suggest_int('cat_iterations', 400, 2000),
                       'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1, log=True),
                       'depth': trial.suggest_int('cat_depth', 4, 10),
                       'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1e-8, 10.0, log=True),
                       'random_state': RANDOM_STATE, 'verbose': 0}
    estimators = [('lgbm', lgb.LGBMRegressor(**lgbm_params)), ('xgb', xgb.XGBRegressor(**xgb_params)),
                  ('catboost', cb.CatBoostRegressor(**catboost_params))]
    stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=3, n_jobs=-1)
    score = cross_val_score(stacking_model, X_processed, y_target, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return -np.mean(score)


# --- 4. ГЛАВНЫЙ СКРИПТ ---
def main():
    # --- ШАГ 1: Загрузка данных ---
    print("--- Шаг 1/5: Загрузка и предварительная обработка данных ---")
    try:
        df = pd.read_csv(FILE_PATH, decimal=',', sep=';', header=None, usecols=COLUMN_MAPPING.keys(),
                         skiprows=1).rename(columns=COLUMN_MAPPING);
        df.dropna(subset=[TARGET_COLUMN], inplace=True)
        X_raw = df[FEATURE_COLUMNS];
        y_full = df[TARGET_COLUMN]
    except Exception as e:
        print(f"❌ ОШИБКА ЗАГРУЗКИ ДАННЫХ: {e}"); return

    # --- ШАГ 2: Запуск Optuna ---
    print(f"\n--- Шаг 2/5: Запуск Optuna ({N_TRIALS_OPTUNA} шагов) с k={N_CLUSTERS} ---")
    print(f"  Результаты будут сохраняться в: {OPTUNA_STORAGE_NAME.replace('sqlite:///', '')}")
    print("  Процесс можно прервать и возобновить в любой момент.")
    temp_preprocessor = FinalModel({}, {}, {}, n_clusters=N_CLUSTERS)
    X_for_optuna = temp_preprocessor._add_cluster_features(temp_preprocessor._log_transform(X_raw), is_fit=True)

    # === ИЗМЕНЕНИЕ: СОЗДАЕМ ИССЛЕДОВАНИЕ С ПОДДЕРЖКОЙ БД И ВОЗОБНОВЛЕНИЯ ===
    study = optuna.create_study(
        direction='minimize',
        storage=OPTUNA_STORAGE_NAME,
        study_name=OPTUNA_STUDY_NAME,
        load_if_exists=True  # ГЛАВНАЯ СТРОКА: продолжить, если исследование уже существует
    )

    study.optimize(lambda trial: objective(trial, X_for_optuna, y_full), n_trials=N_TRIALS_OPTUNA, n_jobs=-1)

    print("\n✅ Поиск параметров завершен!");
    print(f"  Лучшее значение MSE: {study.best_value:.4f}")
    best_params = study.best_params
    LGBM_BEST_PARAMS = {k.replace('lgbm_', ''): v for k, v in best_params.items() if k.startswith('lgbm_')};
    XGB_BEST_PARAMS = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')};
    CATBOOST_BEST_PARAMS = {k.replace('cat_', ''): v for k, v in best_params.items() if k.startswith('cat_')}
    LGBM_BEST_PARAMS.update({'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1});
    XGB_BEST_PARAMS.update({'random_state': RANDOM_STATE, 'n_jobs': -1});
    CATBOOST_BEST_PARAMS.update({'random_state': RANDOM_STATE, 'verbose': 0})

    # --- ШАГ 3: Финальная оценка качества ---
    print(f"\n--- Шаг 3/5: Оценка качества модели на отложенной выборке ({TEST_SIZE_FOR_VALIDATION * 100}%) ---")
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_full, test_size=TEST_SIZE_FOR_VALIDATION,
                                                        random_state=RANDOM_STATE)
    validation_model = FinalModel(LGBM_BEST_PARAMS, XGB_BEST_PARAMS, CATBOOST_BEST_PARAMS, N_CLUSTERS)
    print("  Обучение модели с лучшими параметрами на 80% данных...");
    validation_model.fit(X_train, y_train)
    y_pred = validation_model.predict(X_test);
    print("  ✅ Предсказания для тестовой выборки созданы.")
    rmse = np.sqrt(mean_squared_error(y_test, y_pred));
    mae = mean_absolute_error(y_test, y_pred);
    mape = mean_absolute_percentage_error(y_test, y_pred);
    r2 = r2_score(y_test, y_pred)
    print(
        "\n--- Метрики качества на тестовой выборке ---\n" + f"  RMSE: {rmse:.4f}\n  MAE:  {mae:.4f}\n  MAPE: {mape:.2%}\n  R²:   {r2:.4f}")
    print("\n--- Построение графика 'Реальность vs. Предсказание' ---");
    fig, ax = plt.subplots(figsize=(10, 10));
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', label='Предсказания модели');
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])];
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Идеальная модель (y=x)');
    ax.set_aspect('equal');
    ax.set_xlim(lims);
    ax.set_ylim(lims);
    ax.set_title('График "Реальность vs. Предсказание"', fontsize=16, pad=20);
    ax.set_xlabel('Реальные значения (y_test)', fontsize=12);
    ax.set_ylabel('Предсказанные значения (y_pred)', fontsize=12);
    ax.legend();
    plt.show()

    # --- ШАГ 4: SHAP-анализ ---
    print(f"\n--- Шаг 4/5: Запуск SHAP-анализа и сохранение графиков ---")
    print("  Подготовка данных для SHAP-анализа...");
    X_train_final = validation_model.get_final_features(X_train);
    X_test_final = validation_model.get_final_features(X_test)
    print("  Создание объяснителя SHAP (KernelExplainer)...");
    background_sample = shap.sample(X_train_final, 100, random_state=RANDOM_STATE);
    explainer = shap.KernelExplainer(validation_model.stacking_model.predict, background_sample)
    print("  Расчет SHAP-значений для тестовых данных (может занять несколько минут)...");
    shap_values = explainer.shap_values(X_test_final);
    print("  ✅ Расчет SHAP-значений завершен.")
    if not os.path.exists(SHAP_PLOTS_DIR): os.makedirs(SHAP_PLOTS_DIR); print(
        f"  Создана директория для графиков: '{SHAP_PLOTS_DIR}'")
    print(f"  Сохранение графиков в '{SHAP_PLOTS_DIR}'...");
    shap.summary_plot(shap_values, X_test_final, plot_type="bar", show=False);
    plt.savefig(os.path.join(SHAP_PLOTS_DIR, '1_summary_plot_bar.png'), bbox_inches='tight');
    plt.close()
    shap.summary_plot(shap_values, X_test_final, show=False);
    plt.savefig(os.path.join(SHAP_PLOTS_DIR, '2_summary_plot_beeswarm.png'), bbox_inches='tight');
    plt.close()
    shap.force_plot(explainer.expected_value, shap_values[0, :], X_test_final.iloc[0, :], matplotlib=True, show=False);
    plt.savefig(os.path.join(SHAP_PLOTS_DIR, '3_force_plot_single.png'), bbox_inches='tight');
    plt.close()
    shap.dependence_plot('cluster_id', shap_values, X_test_final, interaction_index=None, show=False);
    plt.savefig(os.path.join(SHAP_PLOTS_DIR, f'4_dependence_plot_cluster_id.png'), bbox_inches='tight');
    plt.close()
    print("  ✅ Все 4 SHAP-графика успешно сохранены!")

    # --- ШАГ 5: Сохранение итогового отчета ---
    print(f"\n--- Шаг 5/5: Сохранение результатов в файл '{RESULTS_FILE_PATH}' ---")
    try:
        with open(RESULTS_FILE_PATH, 'w', encoding='utf-8') as f:
            f.write("====================================================\n");
            f.write("===   РЕЗУЛЬТАТЫ ИССЛЕДОВАТЕЛЬСКОГО ПАЙПЛАЙНА   ===\n");
            f.write(f"====================================================\n\n")
            f.write(f"--- Общая информация ---\n");
            f.write(f"  Имя исследования Optuna: {OPTUNA_STUDY_NAME}\n");
            f.write(f"  Количество кластеров (N_CLUSTERS): {N_CLUSTERS}\n");
            f.write(f"  Количество шагов Optuna (N_TRIALS_OPTUNA): {N_TRIALS_OPTUNA}\n\n")
            f.write("--- Лучшие гиперпараметры, найденные Optuna ---\n");
            for param, value in best_params.items(): f.write(f"  {param}: {value}\n")
            f.write("\n\n");
            f.write(f"--- Метрики качества на тестовой выборке ({TEST_SIZE_FOR_VALIDATION * 100}%) ---\n")
            f.write(f"  R² (Коэффициент детерминации): {r2:.4f}\n");
            f.write(f"  RMSE (Root Mean Squared Error): {rmse:.4f}\n")
            f.write(f"  MAE (Mean Absolute Error):      {mae:.4f}\n");
            f.write(f"  MAPE (Mean Abs. Perc. Error):   {mape:.2%}\n")
        print(f"✅ Результаты успешно сохранены в '{RESULTS_FILE_PATH}'.")
    except Exception as e:
        print(f"❌ Не удалось сохранить файл с результатами: {e}")

    print("\n--- Исследовательский пайплайн полностью завершен ---")


if __name__ == "__main__":
    main()