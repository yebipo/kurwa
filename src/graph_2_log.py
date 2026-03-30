# =========================================================================================
# ===     СКРИПТ V10: ИССЛЕДОВАТЕЛЬСКИЙ ПАЙПЛАЙН С ОЦЕНКОЙ НА HOLD-OUT        ===
# ===  (Лог-признаки -> Optuna -> оценка на 80/20 -> ЗАВЕРШЕНИЕ)                 ===
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

# Глобальные настройки
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
plt.style.use('seaborn-v0_8-whitegrid')

# --- 1. КОНФИГУРАЦИЯ ---
FILE_PATH = 'output_adequate.csv'
RANDOM_STATE = 42
N_TRIALS_OPTUNA = 70
TEST_SIZE_FOR_VALIDATION = 0.2
N_CLUSTERS = 4

COLUMN_MAPPING = {1: 'feature_1', 3: 'feature_3', 5: 'target'}
FEATURE_COLUMNS = ['feature_1', 'feature_3']
TARGET_COLUMN = 'target'


# --- 2. КЛАСС ФИНАЛЬНОЙ МОДЕЛИ (без изменений) ---
class FinalModel:
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
            estimators=estimators, final_estimator=LinearRegression(), cv=3, n_jobs=-1, verbose=0
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
        if X_raw.columns.tolist() != self.feature_names_in_:
            raise ValueError(
                f"Неверные имена колонок! Модель обучалась на {self.feature_names_in_}, а получила {X_raw.columns.tolist()}")
        X_log = self._log_transform(X_raw)
        X_final = self._add_cluster_features(X_log, is_fit=False)
        return self.stacking_model.predict(X_final)


# --- 3. ФУНКЦИЯ ДЛЯ OPTUNA (без изменений) ---
def objective(trial, X_processed, y_target):
    lgbm_params = {
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 400, 2000),
        'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 150),
        'max_depth': trial.suggest_int('lgbm_max_depth', 5, 20),
        'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lgbm_lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lgbm_lambda_l2', 1e-8, 10.0, log=True), 'random_state': RANDOM_STATE,
        'n_jobs': -1, 'verbose': -1}
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 400, 2000),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0), 'random_state': RANDOM_STATE,
        'n_jobs': -1}
    catboost_params = {
        'iterations': trial.suggest_int('cat_iterations', 400, 2000),
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
    print("--- Шаг 1/3: Загрузка и предварительная обработка данных ---")
    try:
        df = pd.read_csv(FILE_PATH, decimal=',', sep=';', header=None, usecols=COLUMN_MAPPING.keys(),
                         skiprows=1).rename(columns=COLUMN_MAPPING)
        df.dropna(subset=[TARGET_COLUMN], inplace=True)
        X_raw = df[FEATURE_COLUMNS]
        y_full = df[TARGET_COLUMN]
    except Exception as e:
        print(f"❌ ОШИБКА ЗАГРУЗКИ ДАННЫХ: {e}");
        return

    # --- ШАГ 2: Запуск поиска параметров с Optuna ---
    print(f"\n--- Шаг 2/3: Запуск Optuna ({N_TRIALS_OPTUNA} шагов) с k={N_CLUSTERS} ---")
    print("Это может занять значительное время...")

    temp_preprocessor = FinalModel({}, {}, {}, n_clusters=N_CLUSTERS)
    X_for_optuna = temp_preprocessor._add_cluster_features(temp_preprocessor._log_transform(X_raw), is_fit=True)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_for_optuna, y_full), n_trials=N_TRIALS_OPTUNA, n_jobs=1)

    print("\n✅ Поиск параметров завершен!")
    print(f"  Лучшее значение MSE на кросс-валидации: {study.best_value:.4f}")
    best_params = study.best_params
    LGBM_BEST_PARAMS = {k.replace('lgbm_', ''): v for k, v in best_params.items() if k.startswith('lgbm_')}
    XGB_BEST_PARAMS = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
    CATBOOST_BEST_PARAMS = {k.replace('cat_', ''): v for k, v in best_params.items() if k.startswith('cat_')}
    LGBM_BEST_PARAMS.update({'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1})
    XGB_BEST_PARAMS.update({'random_state': RANDOM_STATE, 'n_jobs': -1})
    CATBOOST_BEST_PARAMS.update({'random_state': RANDOM_STATE, 'verbose': 0})

    # --- ШАГ 3: Финальная оценка качества на отложенной выборке ---
    print(f"\n--- Шаг 3/3: Оценка качества модели на отложенной выборке ({TEST_SIZE_FOR_VALIDATION * 100}%) ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y_full, test_size=TEST_SIZE_FOR_VALIDATION, random_state=RANDOM_STATE
    )
    validation_model = FinalModel(LGBM_BEST_PARAMS, XGB_BEST_PARAMS, CATBOOST_BEST_PARAMS, N_CLUSTERS)
    print("  Обучение модели с лучшими параметрами на 80% данных...")
    validation_model.fit(X_train, y_train)
    y_pred = validation_model.predict(X_test)
    print("  ✅ Предсказания для тестовой выборки созданы.")

    # Расчет метрик
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("\n--- Метрики качества на тестовой выборке ---")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"  MAE (Mean Absolute Error):      {mae:.4f}")
    print(f"  MAPE (Mean Abs. Perc. Error):   {mape:.2%}")
    print(f"  R² (Коэффициент детерминации):  {r2:.4f}")

    # Визуализация
    print("\n--- Построение графика 'Реальность vs. Предсказание' ---")
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', label='Предсказания модели')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Идеальная модель (y=x)')
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_title('График "Реальность vs. Предсказание" для лучшей модели', fontsize=16, pad=20)
    ax.set_xlabel('Реальные значения (y_test)', fontsize=12)
    ax.set_ylabel('Предсказанные значения (y_pred)', fontsize=12)
    ax.legend()
    plt.show()

    # --- КОНЕЦ ---
    print("\n--- Исследовательский пайплайн полностью завершен ---")


if __name__ == "__main__":
    main()