# =========================================================================
# ===    СКРИПТ ДЛЯ ПРОДВИНУТОЙ НАСТРОЙКИ (V8.0 - БЕЗОПАСНЫЙ СТЕКИНГ)    ===
# =========================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
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

# Параметры для нашей ЛУЧШЕЙ одиночной модели (ЧЕМПИОН)
BEST_LGBM_PARAMS = {
    'n_estimators': 1200, 'learning_rate': 0.1, 'num_leaves': 45,
    'random_state': RANDOM_STATE, 'n_jobs': -1
}

# --- НОВЫЙ БЛОК: "ЛЕГКИЕ" ВЕРСИИ МОДЕЛЕЙ ДЛЯ СТЕКИНГА ---
# Цель - снизить потребление памяти, сохранив разнообразие
LIGHT_RF_PARAMS = {
    'n_estimators': 100, 'max_depth': 25, 'random_state': RANDOM_STATE, 'n_jobs': -1
}
LIGHT_LGBM_PARAMS = {
    'n_estimators': 400, 'learning_rate': 0.1, 'num_leaves': 31,
    'random_state': RANDOM_STATE, 'n_jobs': -1
}


# -------------------------------------------------------------------------

def load_and_prepare_data(file_path, feature_cols, target_col, test_size, random_state):
    """Загружает данные и подготавливает их."""
    print(f"\n--- Загрузка данных ---")
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
    print(f"✅ Данные разделены на обучающую ({len(X_train)}) и тестовую ({len(X_test)}) выборки.")
    return X_train, X_test, y_train, y_test


def evaluate_all_metrics(y_true, y_pred, model_name=""):
    """Считает и выводит все ключевые метрики."""
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

    # --- ЭТАП 1: Обучение модели-чемпиона (наш бенчмарк) ---
    print("\n--- ЭТАП 1: Обучение модели-чемпиона (LightGBM Normal) ---")
    champion_model = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
    champion_model.fit(X_train, y_train)
    y_pred_champion = champion_model.predict(X_test)
    all_results['LightGBM (Чемпион)'] = evaluate_all_metrics(y_test, y_pred_champion, "LightGBM (Чемпион)")

    # --- ЭТАП 2: Стекинг в безопасном режиме ---
    print("\n" + "=" * 50)
    print("--- ЭТАП 2: Стекинг 'легких' моделей (безопасный режим) ---")

    # 1. Определяем "легкие" базовые модели
    light_estimators = [
        ('random_forest', RandomForestRegressor(**LIGHT_RF_PARAMS)),
        ('lightgbm', lgb.LGBMRegressor(**LIGHT_LGBM_PARAMS))
    ]

    # 2. Создаем стекинг с безопасными параметрами
    stacking_regressor = StackingRegressor(
        estimators=light_estimators,
        final_estimator=LinearRegression(),
        cv=3,  # Уменьшенное количество фолдов
        n_jobs=1,  # Строго последовательное выполнение
        verbose=2  # Добавим verbose, чтобы видеть прогресс!
    )

    print("Обучение стекинг-модели (это займет некоторое время)...")
    start_time = time.time()
    stacking_regressor.fit(X_train, y_train)
    stacking_time = time.time() - start_time
    print(f"\nОбучение стекинга завершено за {stacking_time:.2f} сек.")

    y_pred_stacking = stacking_regressor.predict(X_test)
    all_results['Стекинг (легкие модели)'] = evaluate_all_metrics(y_test, y_pred_stacking, "Стекинг (легкие модели)")

    # --- ЭТАП 3: Итоговое сравнение ---
    print("\n" + "=" * 50)
    print("###          СРАВНЕНИЕ ЧЕМПИОНА И СТЕКИНГА        ###")
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


if __name__ == "__main__":
    main()