# ===================================================================================
# ===  СКРИПТ ДЛЯ ПРОДВИНУТОЙ НАСТРОЙКИ (V12.2 - ИСПРАВЛЕНИЕ СТЕКИНГА С XGB) ===
# ===================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import time
import warnings

warnings.filterwarnings('ignore')

# --- 1. КОНФИГУРАЦИЯ ---
FILE_PATH = 'output_adequate.csv'
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CLUSTERS = 8

CHAMPION_LGBM_PARAMS = {
    'n_estimators': 1470, 'learning_rate': 0.088075, 'num_leaves': 57,
    'max_depth': 18, 'min_child_samples': 13, 'lambda_l1': 1.644e-07,
    'lambda_l2': 3.641e-08, 'random_state': RANDOM_STATE, 'n_jobs': -1
}

XGB_PARAMS = {
    'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 7,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': RANDOM_STATE,
    'n_jobs': -1, 'early_stopping_rounds': 50
}
CATBOOST_PARAMS = {
    'iterations': 1000, 'learning_rate': 0.05, 'depth': 7,
    'l2_leaf_reg': 3, 'random_state': RANDOM_STATE, 'verbose': 0
}

# --- НОВЫЙ БЛОК: Параметры для XGBoost ВНУТРИ стекинга ---
# Копируем основные параметры и УДАЛЯЕМ early_stopping_rounds
XGB_PARAMS_FOR_STACKING = XGB_PARAMS.copy()
del XGB_PARAMS_FOR_STACKING['early_stopping_rounds']


# -------------------------------------------------------------------------

def load_and_prepare_data(file_path, feature_cols, target_col, test_size, random_state):
    # Эта функция без изменений
    print(f"\n--- Загрузка и подготовка данных ---")
    try:
        df = pd.read_csv(file_path, decimal=',', sep=';', header=None, usecols=feature_cols + [target_col], skiprows=1)
    except Exception as e:
        print(f"❌ ОШИБКА: {e}");
        return None
    df.dropna(subset=[target_col], inplace=True)
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"✅ Данные разделены.")
    return X_train, X_test, y_train, y_test


def add_cluster_features(X_train, X_test, n_clusters, random_state):
    # Эта функция без изменений
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    X_train_with_cluster = X_train.copy()
    X_train_with_cluster['cluster_id'] = kmeans.fit_predict(X_train_scaled)
    X_test_with_cluster = X_test.copy()
    X_test_with_cluster['cluster_id'] = kmeans.predict(X_test_scaled)
    return X_train_with_cluster, X_test_with_cluster


def evaluate_all_metrics(y_true, y_pred, model_name=""):
    # Эта функция без изменений
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_for_mape = y_true.copy()
    y_true_for_mape[y_true_for_mape == 0] = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / y_true_for_mape)) * 100
    print(f"\n--- Метрики для модели: {model_name} ---")
    print(f"  R-squared: {r2:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.2f}%")
    return {'R-squared': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}


def main():
    all_results = {}

    data = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    if data is None: return
    X_train, X_test, y_train, y_test = data

    print("\nДобавление признака кластеризации (k=8)...")
    X_train_final, X_test_final = add_cluster_features(X_train, X_test, N_CLUSTERS, RANDOM_STATE)

    # --- ЭТАП 1: Индивидуальные заезды Титанов ---

    print("\n--- Обучение LGBM (Чемпион)... ---")
    lgbm_model = lgb.LGBMRegressor(**CHAMPION_LGBM_PARAMS)
    lgbm_model.fit(X_train_final, y_train, categorical_feature=['cluster_id'])
    y_pred_lgbm = lgbm_model.predict(X_test_final)
    all_results['LGBM (Optuna)'] = evaluate_all_metrics(y_test, y_pred_lgbm, "LGBM (Optuna)")

    print("\n--- Обучение XGBoost... ---")
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS, enable_categorical=True)
    X_train_xgb, X_test_xgb = X_train_final.copy(), X_test_final.copy()
    X_train_xgb['cluster_id'] = X_train_xgb['cluster_id'].astype('category')
    X_test_xgb['cluster_id'] = X_test_xgb['cluster_id'].astype('category')
    xgb_model.fit(X_train_xgb, y_train, eval_set=[(X_test_xgb, y_test)], verbose=False)
    y_pred_xgb = xgb_model.predict(X_test_xgb)
    all_results['XGBoost'] = evaluate_all_metrics(y_test, y_pred_xgb, "XGBoost")

    print("\n--- Обучение CatBoost... ---")
    cat_model = cb.CatBoostRegressor(**CATBOOST_PARAMS)
    cat_model.fit(X_train_final, y_train, cat_features=['cluster_id'])
    y_pred_cat = cat_model.predict(X_test_final)
    all_results['CatBoost'] = evaluate_all_metrics(y_test, y_pred_cat, "CatBoost")

    # --- ЭТАП 2: Стекинг Титанов ---
    print("\n" + "=" * 50)
    print("--- ЭТАП 2: Стекинг Титанов (LGBM + XGB + CatBoost) ---")

    # --- ИСПРАВЛЕНИЕ: Используем специальную версию параметров для XGBoost ---
    estimators = [
        ('lgbm', lgb.LGBMRegressor(**CHAMPION_LGBM_PARAMS)),
        ('xgb', xgb.XGBRegressor(**XGB_PARAMS_FOR_STACKING, enable_categorical=True)),
        ('catboost', cb.CatBoostRegressor(**CATBOOST_PARAMS))
    ]

    stacking_regressor = StackingRegressor(
        estimators=estimators, final_estimator=LinearRegression(), cv=3, n_jobs=1, verbose=2
    )

    print("Обучение стекинг-модели (это займет много времени)...")
    stacking_regressor.fit(X_train_final, y_train)

    y_pred_stacking = stacking_regressor.predict(X_test_final)
    all_results['Стекинг Титанов'] = evaluate_all_metrics(y_test, y_pred_stacking, "Стекинг Титанов")

    # --- ЭТАП 3: Финальная таблица ---
    print("\n" + "=" * 50)
    print("###          РЕЗУЛЬТАТЫ БИТВЫ ТИТАНОВ        ###")
    print("=" * 50)

    results_df = pd.DataFrame(all_results).T
    results_df = results_df.sort_values(by='RMSE', ascending=True)

    formatters = {'R-squared': '{:,.6f}'.format, 'MAE': '{:,.6f}'.format, 'RMSE': '{:,.6f}'.format,
                  'MAPE (%)': '{:,.2f}%'.format}
    print(results_df.to_string(formatters=formatters))

    best_model_name = results_df.index[0]
    print(f"\n🏆 Абсолютный победитель: '{best_model_name}'")


if __name__ == "__main__":
    main()