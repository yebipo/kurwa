# =========================================================================
# ===    СКРИПТ ДЛЯ ПРОДВИНУТОЙ НАСТРОЙКИ (V5.0 - ЛОГАРИФМИРОВАНИЕ)      ===
# =========================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import lightgbm as lgb
import time
import warnings
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- 1. КОНФИГУРАЦИЯ ---
FILE_PATH = 'output_adequate.csv'
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Облегченная сетка для RF
RF_PARAM_GRID_EXTENDED = {
    'n_estimators': [400], 'max_depth': [30, 40],
    'min_samples_leaf': [1, 2], 'max_features': [0.7, 'sqrt']
}
# Сетка для LGBM
LGBM_FINETUNE_GRID = {
    'n_estimators': [1000, 1200], 'learning_rate': [0.08, 0.1],
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
    # Эта функция остается без изменений
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


def main():
    results = {}

    # --- ЭТАП 1: Обычное обучение ---
    data = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    if data is None: return
    X_train, X_test, y_train, y_test = data

    best_lgbm_model = run_grid_search(X_train, y_train, lgb.LGBMRegressor(), LGBM_FINETUNE_GRID, "LightGBM",
                                      RANDOM_STATE)
    results['Настроенный LightGBM'] = {'R-squared': best_lgbm_model.score(X_test, y_test)}

    # --- ЭТАП 2: Обучение на логарифмированных данных ---
    print("\n--- ЭТАП 2: Эксперимент с логарифмированием 'y' ---")
    data_log = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE,
                                     log_transform=True)
    if data_log is None: return
    X_train_log, X_test_log, y_train_log, y_test_log = data_log

    # Обучаем ту же модель, но на лог-данных
    best_lgbm_log_model = run_grid_search(X_train_log, y_train_log, lgb.LGBMRegressor(), LGBM_FINETUNE_GRID,
                                          "LightGBM on Log(y)", RANDOM_STATE)

    # 1. Получаем предсказания в логарифмическом масштабе
    y_pred_log = best_lgbm_log_model.predict(X_test_log)
    # 2. Возвращаем их в исходный масштаб
    y_pred_original_scale = np.expm1(y_pred_log)

    # 3. Считаем R-squared, сравнивая ОРИГИНАЛЬНЫЕ y_test и ПРЕОБРАЗОВАННЫЕ ОБРАТНО предсказания
    log_score = r2_score(y_test, y_pred_original_scale)  # ВАЖНО: используем y_test из первого этапа
    results['LGBM + Log(y)'] = {'R-squared': log_score}

    # --- ЭТАП 3: Итоговое сравнение ---
    print("\n==================================================")
    print("###          ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ        ###")
    print("==================================================")
    results_df = pd.DataFrame(results).T.sort_values(by='R-squared', ascending=False)
    print(results_df.to_string(formatters={'R-squared': '{:,.6f}'.format}))

    best_model_name = results_df.index[0]
    best_score = results_df['R-squared'].iloc[0]
    print(f"\n🏆🏆🏆 Лучший подход: '{best_model_name}' с R-squared = {best_score:.6f}")


if __name__ == "__main__":
    main()