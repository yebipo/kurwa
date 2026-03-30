# =====================================================================================
# === СКРИПТ ДЛЯ ПРОДВИНУТОЙ НАСТРОЙКИ (V6.1 - АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ С КОНСТАНТОЙ) ===
# =====================================================================================

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

# Используем лучшие параметры, найденные в предыдущих запусках
BEST_LGBM_PARAMS = {
    'n_estimators': 1200,
    'learning_rate': 0.1,
    'num_leaves': 45,
    'random_state': RANDOM_STATE,
    'n_jobs': -1  # Использовать все доступные ядра
}


# -------------------------------------------------------------------------

def load_and_prepare_data(file_path, feature_cols, target_col, test_size, random_state, use_const_features=False):
    """Загружает данные и, если use_const_features=True, создает новые признаки с константой."""
    print(f"\n--- Загрузка данных (Использовать константу: {use_const_features}) ---")
    try:
        df = pd.read_csv(
            file_path, decimal=',', sep=';', header=None,
            usecols=feature_cols + [target_col], skiprows=1
        )
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        return None

    df.dropna(subset=[target_col], inplace=True)
    X = df[feature_cols].copy()  # Используем .copy() во избежание SettingWithCopyWarning
    y = df[target_col]

    if use_const_features:
        print("   Создание новых признаков с использованием константы 16384...")
        for col in feature_cols:
            new_col_index = col + 100  # Просто чтобы не пересекаться
            X[new_col_index] = X[col] / THE_CONSTANT
        print(f"   Количество признаков увеличено до: {X.shape[1]}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Данные разделены.")
    return X_train, X_test, y_train, y_test


def evaluate_all_metrics(y_true, y_pred, model_name=""):
    """Считает и выводит R2, MAE, RMSE и MAPE."""
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    y_true_for_mape = y_true.copy()
    y_true_for_mape[y_true_for_mape == 0] = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / y_true_for_mape)) * 100

    print(f"\n--- Метрики для модели: {model_name} ---")
    print(f"  R-squared: {r2:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAPE: {mape:.2f}%")

    return {'R-squared': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}


def main():
    all_results = {}

    # --- ЭТАП 1: Обучение на базовых признаках ---
    data_normal = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
                                        use_const_features=False)
    if data_normal is None: return
    X_train, X_test, y_train, y_test = data_normal

    print("\n--- Обучение на 4 базовых признаках ---")
    model_normal = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
    model_normal.fit(X_train, y_train)
    y_pred_normal = model_normal.predict(X_test)
    all_results['LGBM (4 признака)'] = evaluate_all_metrics(y_test, y_pred_normal, "LGBM (4 признака)")

    # --- ЭТАП 2: Обучение на расширенных признаках с константой ---
    print("\n" + "=" * 50)
    data_const = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
                                       use_const_features=True)
    if data_const is None: return
    X_train_const, X_test_const, y_train_const, _ = data_const

    print("\n--- Обучение на 8 признаках (с константой) ---")
    model_const = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
    model_const.fit(X_train_const, y_train_const)
    y_pred_const = model_const.predict(X_test_const)
    all_results['LGBM (8 признаков с константой)'] = evaluate_all_metrics(y_test, y_pred_const,
                                                                          "LGBM (8 признаков с константой)")

    # --- НОВЫЙ БЛОК: Анализ важности признаков ---
    print("\n--- Анализ важности признаков для модели с 8 признаками ---")
    feature_importances = pd.DataFrame({
        'feature': X_train_const.columns,
        'importance': model_const.feature_importances_
    }).sort_values('importance', ascending=False)
    print("Важность каждого признака (количество раз, когда признак использовался для разделения):")
    print(feature_importances.to_string(index=False))

    # --- ЭТАП 3: Итоговое сравнение ---
    print("\n" + "=" * 50)
    print("###          ИТОГОВОЕ СРАВНЕНИЕ ЭКСПЕРИМЕНТОВ        ###")
    print("=" * 50)

    results_df = pd.DataFrame(all_results).T
    results_df = results_df.sort_values(by='RMSE', ascending=True)

    formatters = {
        'R-squared': '{:,.6f}'.format, 'MAE': '{:,.6f}'.format,
        'RMSE': '{:,.6f}'.format, 'MAPE (%)': '{:,.2f}%'.format
    }
    print(results_df.to_string(formatters=formatters))

    best_model_name = results_df.index[0]
    print(f"\n🏆 Лучший подход: '{best_model_name}'")
    print(
        "   Вывод: Добавление отмасштабированных признаков не дает новой информации для древовидных моделей, таких как LightGBM.")


if __name__ == "__main__":
    main()