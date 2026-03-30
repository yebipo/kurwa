# ===================================================================================
# === СКРИПТ ДЛЯ ПРОДВИНУТОЙ НАСТРОЙКИ (V11.1 - БЫСТРЫЙ ТЕСТ С OPTUNA) ===
# ===================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import optuna
import time
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- 1. КОНФИГУРАЦИЯ ---
FILE_PATH = 'output_adequate.csv'
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_CLUSTERS = 8  # Оптимальное k из прошлого эксперимента

# --- ИЗМЕНЕНИЕ: Сокращаем количество попыток для быстрого теста ---
N_TRIALS_OPTUNA = 30

# Параметры нашего текущего чемпиона для сравнения
CHAMPION_LGBM_PARAMS = {
    'n_estimators': 1200, 'learning_rate': 0.1, 'num_leaves': 45,
    'random_state': RANDOM_STATE, 'n_jobs': -1
}


# -------------------------------------------------------------------------

def load_and_prepare_data(file_path, feature_cols, target_col, test_size, random_state):
    print(f"\n--- Загрузка и подготовка данных ---")
    try:
        df = pd.read_csv(
            file_path, decimal=',', sep=';', header=None,
            usecols=feature_cols + [target_col], skiprows=1
        )
    except Exception as e:
        print(f"❌ ОШИБКА: {e}");
        return None

    df.dropna(subset=[target_col], inplace=True)
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Данные разделены.")
    return X_train, X_test, y_train, y_test


def add_cluster_features(X_train, X_test, n_clusters, random_state):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')

    X_train_with_cluster = X_train.copy()
    X_train_with_cluster['cluster_id'] = kmeans.fit_predict(X_train_scaled)

    X_test_with_cluster = X_test.copy()
    X_test_with_cluster['cluster_id'] = kmeans.predict(X_test_scaled)

    return X_train_with_cluster, X_test_with_cluster


# --- ЦЕЛЕВАЯ ФУНКЦИЯ ДЛЯ OPTUNA (упрощенная) ---
def objective(trial, X, y):
    # 1. Задаем диапазоны только для самых важных параметров
    param = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 800, 1500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 80),
        'max_depth': trial.suggest_int('max_depth', 7, 25),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 5.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 5.0, log=True),
        'verbose': -1,
        'n_jobs': -1,
        'seed': RANDOM_STATE,
    }

    # 2. Используем кросс-валидацию для надежной оценки
    kf = KFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  # Уменьшили до 3-х фолдов
    rmse_scores = []

    X_values = X.values
    y_values = y.values

    for train_index, val_index in kf.split(X_values):
        X_train_fold, X_val_fold = X_values[train_index], X_values[val_index]
        y_train_fold, y_val_fold = y_values[train_index], y_values[val_index]

        model = lgb.LGBMRegressor(**param)
        model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(50, verbose=False)])  # Уменьшили patience для early_stopping

        preds = model.predict(X_val_fold)
        rmse = np.sqrt(mean_squared_error(y_val_fold, preds))
        rmse_scores.append(rmse)

    # 3. Возвращаем среднее значение RMSE
    return np.mean(rmse_scores)


def evaluate_all_metrics(y_true, y_pred, model_name=""):
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
    data = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    if data is None: return
    X_train, X_test, y_train, y_test = data

    print("\nДобавление признака кластеризации (k=8)...")
    X_train_final, X_test_final = add_cluster_features(X_train, X_test, N_CLUSTERS, RANDOM_STATE)

    # --- ЭТАП 1: Поиск лучших параметров с Optuna ---
    print("\n--- ЭТАП 1: Запуск оптимизации с Optuna ---")
    print(f"Будет выполнено {N_TRIALS_OPTUNA} попыток...")

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train_final, y_train), n_trials=N_TRIALS_OPTUNA,
                   show_progress_bar=True)

    print("\nОптимизация завершена!")
    print(f"Лучшее значение RMSE на валидации: {study.best_value:.6f}")
    print("Лучшие найденные параметры:")
    print(study.best_params)

    # --- ЭТАП 2: Сравнение с чемпионом ---
    print("\n" + "=" * 50)
    print("--- ЭТАП 2: Сравнение моделей на тестовых данных ---")

    print("\nОбучение модели с параметрами от Optuna...")
    optuna_params = study.best_params
    optuna_params['random_state'] = RANDOM_STATE
    optuna_params['n_jobs'] = -1
    optuna_model = lgb.LGBMRegressor(**optuna_params)
    optuna_model.fit(X_train_final, y_train, categorical_feature=['cluster_id'])
    y_pred_optuna = optuna_model.predict(X_test_final)
    optuna_metrics = evaluate_all_metrics(y_test, y_pred_optuna, "LGBM (Optuna)")

    print("\nОбучение модели-чемпиона (GridSearchCV)...")
    champion_model = lgb.LGBMRegressor(**CHAMPION_LGBM_PARAMS)
    champion_model.fit(X_train_final, y_train, categorical_feature=['cluster_id'])
    y_pred_champion = champion_model.predict(X_test_final)
    champion_metrics = evaluate_all_metrics(y_test, y_pred_champion, "LGBM (GridSearchCV)")

    # --- ЭТАП 3: Финальная таблица ---
    print("\n" + "=" * 50)
    print("###          OPTUNA vs GRIDSEARCHCV        ###")
    print("=" * 50)

    results_df = pd.DataFrame({
        "LGBM (Optuna)": optuna_metrics,
        "LGBM (GridSearchCV)": champion_metrics
    }).T

    formatters = {
        'R-squared': '{:,.6f}'.format, 'MAE': '{:,.6f}'.format,
        'RMSE': '{:,.6f}'.format, 'MAPE (%)': '{:,.2f}%'.format
    }
    print(results_df.to_string(formatters=formatters))

    best_model_name = results_df.sort_values(by='RMSE', ascending=True).index[0]
    print(f"\n🏆 Победитель: '{best_model_name}'")


if __name__ == "__main__":
    main()