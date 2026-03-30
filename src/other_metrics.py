# =========================================================================
# ===    СКРИПТ ДЛЯ ПРОДВИНУТОЙ НАСТРОЙКИ (V5.1 - КОМПЛЕКСНАЯ ОЦЕНКА)    ===
# =========================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import time
import warnings
import joblib

warnings.filterwarnings('ignore')

# --- 1. КОНФИГУРАЦИЯ ---
FILE_PATH = 'output_adequate.csv'
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Сетка для точечной настройки LightGBM (из прошлого успешного запуска)
LGBM_FINETUNE_GRID = {
    'n_estimators': [1000, 1200],
    'learning_rate': [0.08, 0.1],
    'num_leaves': [45, 50, 55]
}


# -------------------------------------------------------------------------

def load_and_prepare_data(file_path, feature_cols, target_col, test_size, random_state, log_transform=False):
    """Загружает данные. Если log_transform=True, логарифмирует целевую переменную."""
    print(f"\n--- Загрузка данных (Log Transform: {log_transform}) ---")
    try:
        df = pd.read_csv(
            file_path, decimal=',', sep=';', header=None,
            usecols=feature_cols + [target_col], skiprows=1
        )
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        return None

    df.dropna(subset=[target_col], inplace=True)
    X = df[feature_cols]
    y = df[target_col]

    if log_transform:
        y = np.log1p(y)  # y_log = log(1+y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Данные разделены.")
    return X_train, X_test, y_train, y_test


def run_grid_search(X_train, y_train, estimator, param_grid, model_name, random_state):
    """Выполняет GridSearchCV и возвращает лучшую модель."""
    print(f"--- Поиск лучших параметров для {model_name} (GridSearchCV) ---")
    try:
        estimator.set_params(random_state=random_state)
    except ValueError:
        pass

    grid_search = GridSearchCV(estimator=estimator, param_grid=param_grid, cv=3, n_jobs=4, verbose=2, scoring='r2')

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time

    print(f"\nПоиск для {model_name} завершен за {search_time:.2f} сек.")
    print(f"🏆 Лучшие параметры: {grid_search.best_params_}")
    print(f"   Лучший R-squared на валидации: {grid_search.best_score_:.6f}\n")
    return grid_search.best_estimator_


def evaluate_all_metrics(y_true, y_pred, model_name=""):
    """Считает и выводит R2, MAE, RMSE и MAPE."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # Считаем MAPE аккуратно, чтобы избежать деления на ноль, если y_true содержит 0
    y_true_for_mape = y_true.copy()
    y_true_for_mape[y_true_for_mape == 0] = 1e-6  # Заменяем 0 на очень маленькое число
    mape = np.mean(np.abs((y_true - y_pred) / y_true_for_mape)) * 100

    print(f"\n--- Метрики для модели: {model_name} ---")
    print(f"  R-squared: {r2:.6f}")
    print(f"  MAE (Средняя абсолютная ошибка): {mae:.6f}")
    print(f"  RMSE (Корень из среднеквадратичной ошибки): {rmse:.6f}")
    print(f"  MAPE (Средняя абсолютная процентная ошибка): {mape:.2f}%")

    return {'R-squared': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}


def main():
    all_results = {}

    # --- ЭТАП 1: Обычное обучение ---
    data_normal = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    if data_normal is None: return
    X_train, X_test, y_train, y_test = data_normal

    best_lgbm_normal = run_grid_search(X_train, y_train, lgb.LGBMRegressor(), LGBM_FINETUNE_GRID, "LightGBM (Normal)",
                                       RANDOM_STATE)
    y_pred_normal = best_lgbm_normal.predict(X_test)
    all_results['LightGBM (Normal)'] = evaluate_all_metrics(y_test, y_pred_normal, "LightGBM (Normal)")

    # --- ЭТАП 2: Обучение на логарифмированных данных ---
    print("\n" + "=" * 50)
    data_log = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
                                     log_transform=True)
    if data_log is None: return
    X_train_log, X_test_log, y_train_log, y_test_log = data_log

    best_lgbm_log = run_grid_search(X_train_log, y_train_log, lgb.LGBMRegressor(), LGBM_FINETUNE_GRID,
                                    "LightGBM on Log(y)", RANDOM_STATE)

    # Делаем предсказания и возвращаем их в исходный масштаб
    y_pred_log = best_lgbm_log.predict(X_test_log)
    y_pred_original_scale = np.expm1(y_pred_log)

    # Оцениваем на исходных данных
    all_results['LightGBM + Log(y)'] = evaluate_all_metrics(y_test, y_pred_original_scale, "LightGBM + Log(y)")

    # --- ЭТАП 3: Итоговое сравнение ---
    print("\n" + "=" * 50)
    print("###          ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ ПО ВСЕМ МЕТРИКАМ        ###")
    print("=" * 50)

    results_df = pd.DataFrame(all_results).T
    # Сортируем по R-squared, но можно выбрать и другую метрику, например, RMSE
    results_df = results_df.sort_values(by='R-squared', ascending=False)

    # Форматирование для красивого вывода
    formatters = {
        'R-squared': '{:,.6f}'.format,
        'MAE': '{:,.6f}'.format,
        'RMSE': '{:,.6f}'.format,
        'MAPE (%)': '{:,.2f}%'.format
    }
    print(results_df.to_string(formatters=formatters))

    # Определение лучшей модели по RMSE (т.к. она более чувствительна к большим ошибкам)
    best_model_by_rmse = results_df.sort_values(by='RMSE', ascending=True).index[0]
    print(f"\n🏆 Лучшая модель по метрике RMSE: '{best_model_by_rmse}'")


if __name__ == "__main__":
    main()