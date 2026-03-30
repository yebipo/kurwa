# =========================================================================
# ===    СКРИПТ ДЛЯ ПРОДВИНУТОЙ НАСТРОЙКИ (V7.0 - РЕШАЮЩИЙ ЭКСПЕРИМЕНТ)   ===
# =========================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import lightgbm as lgb
import time
import warnings

warnings.filterwarnings('ignore')

# --- 1. КОНФИГУРАЦИЯ ---
FILE_PATH = 'output_adequate.csv'
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42
THE_CONSTANT = 16384.0  # Ваша константа 2^14

BEST_LGBM_PARAMS = {
    'n_estimators': 1200, 'learning_rate': 0.1, 'num_leaves': 45,
    'random_state': RANDOM_STATE, 'n_jobs': -1
}


# -------------------------------------------------------------------------

def load_and_prepare_data(file_path, feature_cols, target_col, test_size, random_state, engineering_mode='none'):
    """
    Загружает данные и создает новые признаки в зависимости от engineering_mode.
    Modes: 'none', 'modulo', 'floor', 'all'
    """
    print(f"\n--- Загрузка данных (Режим инжиниринга: {engineering_mode}) ---")
    try:
        df = pd.read_csv(
            file_path, decimal=',', sep=';', header=None,
            usecols=feature_cols + [target_col], skiprows=1
        )
    except Exception as e:
        print(f"❌ ОШИБКА: {e}");
        return None

    df.dropna(subset=[target_col], inplace=True)
    X = df[feature_cols].copy()
    y = df[target_col]

    if engineering_mode == 'modulo' or engineering_mode == 'all':
        print("   Создание признаков с остатком от деления (% K)...")
        for col in feature_cols:
            X[f'{col}_mod_K'] = X[col] % THE_CONSTANT

    if engineering_mode == 'floor' or engineering_mode == 'all':
        print("   Создание признаков с целочисленным делением (// K)...")
        for col in feature_cols:
            X[f'{col}_floor_K'] = X[col] // THE_CONSTANT

    if engineering_mode != 'none':
        print(f"   Итоговое количество признаков: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Данные разделены.")
    return X_train, X_test, y_train, y_test


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


def run_experiment(X_train, X_test, y_train, y_test, model_name):
    """Обучает модель и возвращает метрики."""
    print(f"\n--- Обучение модели: {model_name} ---")
    model = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return evaluate_all_metrics(y_test, y_pred, model_name)


def main():
    all_results = {}

    # Загружаем y_test один раз для всех экспериментов, чтобы сравнение было честным
    _, _, _, y_test = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    if y_test is None: return

    # --- Эксперименты ---
    modes = ['none', 'modulo', 'floor', 'all']
    friendly_names = {
        'none': 'LGBM (4 базовых признака)',
        'modulo': 'LGBM (8 признаков, с % K)',
        'floor': 'LGBM (8 признаков, с // K)',
        'all': 'LGBM (12 признаков, все вместе)'
    }

    for mode in modes:
        data = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
                                     engineering_mode=mode)
        if data is None: continue
        X_train, X_test, y_train, _ = data
        model_name = friendly_names[mode]
        all_results[model_name] = run_experiment(X_train, X_test, y_train, y_test, model_name)

    # --- Итоговое сравнение ---
    print("\n" + "=" * 50)
    print("###          РЕЗУЛЬТАТЫ РЕШАЮЩЕГО ЭКСПЕРИМЕНТА        ###")
    print("=" * 50)

    results_df = pd.DataFrame(all_results).T
    results_df = results_df.sort_values(by='RMSE', ascending=True)

    formatters = {
        'R-squared': '{:,.6f}'.format, 'MAE': '{:,.6f}'.format,
        'RMSE': '{:,.6f}'.format, 'MAPE (%)': '{:,.2f}%'.format
    }
    print(results_df.to_string(formatters=formatters))

    best_model_name = results_df.index[0]
    print(f"\n🏆🏆🏆 Лучший подход: '{best_model_name}'")


if __name__ == "__main__":
    main()