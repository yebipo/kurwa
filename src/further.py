# ===================================================================================
# ===  СКРИПТ ДЛЯ ПРОДВИНУТОЙ НАСТРОЙКИ (V4.2 - С ЗАЩИТОЙ ОТ ЗАВИСАНИЯ СТЕКИНГА) ===
# ===================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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

# Более "легкая" сетка для RandomForest для предотвращения ошибок памяти
RF_PARAM_GRID_EXTENDED = {
    'n_estimators': [400],
    'max_depth': [30, 40],
    'min_samples_leaf': [1, 2],
    'max_features': [0.7, 'sqrt']
}

# Сетка для точечной настройки LightGBM
LGBM_FINETUNE_GRID = {
    'n_estimators': [1000, 1200],
    'learning_rate': [0.08, 0.1],
    'num_leaves': [45, 50, 55],
    'reg_alpha': [0.1, 0.2],
    'reg_lambda': [0.1, 0.2]
}


# -------------------------------------------------------------------------

def load_and_prepare_data(file_path, feature_cols, target_col, test_size, random_state):
    print("--- ЭТАП 1: Загрузка и подготовка данных ---")
    try:
        all_required_cols = feature_cols + [target_col]
        df = pd.read_csv(
            file_path, decimal=',', sep=';', header=None,
            usecols=all_required_cols, skiprows=1
        )
        print(f"✅ Данные успешно загружены, первая строка пропущена. Анализируется {df.shape[0]} строк.")
    except Exception as e:
        print(f"❌ ОШИБКА при загрузке данных: {e}")
        return None

    initial_rows = len(df)
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
    df.dropna(subset=[target_col], inplace=True)
    if len(df) < initial_rows:
        print(f"⚠️  Удалено {initial_rows - len(df)} строк с нечисловыми/пустыми значениями в целевой колонке.")

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Данные разделены на обучающую ({len(X_train)}) и тестовую ({len(X_test)}) выборки.\n")
    return X_train, X_test, y_train, y_test


def run_grid_search(X_train, y_train, estimator, param_grid, model_name, random_state):
    print(f"--- Поиск лучших параметров для {model_name} (GridSearchCV) ---")
    try:
        estimator.set_params(random_state=random_state)
    except ValueError:
        pass

    grid_search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=3,
        n_jobs=4,  # Ограничиваем количество параллельных процессов
        verbose=2,
        scoring='r2'
    )
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    search_time = time.time() - start_time

    print(f"\nПоиск по сетке для {model_name} завершен за {search_time:.2f} секунд.")
    print(f"🏆 Лучшие найденные параметры: {grid_search.best_params_}")
    print(f"   Лучший R-squared на валидации: {grid_search.best_score_:.6f}\n")
    return grid_search.best_estimator_


def main():
    """Главная функция, запускающая весь процесс."""
    prepared_data = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    if prepared_data is None: return
    X_train, X_test, y_train, y_test = prepared_data
    if len(X_train) == 0:
        print("❌ ОШИБКА: После очистки не осталось данных для обучения.")
        return

    results = {}

    # --- ЭТАП 2: Настройка основных моделей ---
    print("\n--- ЭТАП 2: Настройка основных моделей ---")
    best_rf_model = run_grid_search(
        X_train, y_train, RandomForestRegressor(), RF_PARAM_GRID_EXTENDED, "RandomForest", RANDOM_STATE
    )
    results['Настроенный RandomForest'] = {'R-squared': best_rf_model.score(X_test, y_test),
                                           'Время обучения (сек)': '-'}

    best_lgbm_model = run_grid_search(
        X_train, y_train, lgb.LGBMRegressor(), LGBM_FINETUNE_GRID, "LightGBM (Fine-Tune)", RANDOM_STATE
    )
    results['Настроенный LightGBM'] = {'R-squared': best_lgbm_model.score(X_test, y_test), 'Время обучения (сек)': '-'}

    # --- ЭТАП 3: Анализ ошибок лучшей модели ---
    print("\n--- ЭТАП 3: Анализ ошибок лучшей модели (LightGBM) ---")
    y_pred_lgbm = best_lgbm_model.predict(X_test)
    residuals = y_test - y_pred_lgbm
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x=y_test, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Анализ остатков: y_test vs. Ошибки (residuals)', fontsize=16)
    plt.xlabel('Реальные значения (y_test)', fontsize=12)
    plt.ylabel('Ошибка предсказания (Residuals)', fontsize=12)
    plt.grid(True)
    plt.show()

    # --- ЭТАП 4: Стекинг моделей ---
    print("\n--- ЭТАП 4: Стекинг моделей (RandomForest + LightGBM) ---")
    estimators = [('random_forest', best_rf_model), ('lightgbm', best_lgbm_model)]

    # --- ИЗМЕНЕНИЕ: Добавлены параметры для предотвращения зависания ---
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=LinearRegression(),
        cv=3,  # Уменьшено количество фолдов для экономии памяти
        n_jobs=1  # Запрещен внутренний параллелизм для стабильности
    )

    print("Обучение стекинг-модели (в режиме экономии памяти)...")
    start_time = time.time()
    stacking_regressor.fit(X_train, y_train)
    stacking_time = time.time() - start_time
    stacking_score = stacking_regressor.score(X_test, y_test)
    print(f"   Обучение завершено за {stacking_time:.2f} сек.")
    results['Стекинг (RF+LGBM)'] = {'R-squared': stacking_score, 'Время обучения (сек)': stacking_time}

    # --- ЭТАП 5: Итоговое сравнение и сохранение ---
    print("\n==================================================")
    print("###          ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ        ###")
    print("==================================================")

    results_df = pd.DataFrame(results).T.sort_values(by='R-squared', ascending=False)
    formatters = {'R-squared': '{:,.6f}'.format,
                  'Время обучения (сек)': lambda x: f"{x:,.2f}" if pd.notna(x) and x != '-' else '-'}
    print(results_df.to_string(formatters=formatters))

    best_model_name = results_df.index[0]
    best_score = results_df['R-squared'].iloc[0]
    print(f"\n🏆🏆🏆 Лучший подход: '{best_model_name}' с R-squared = {best_score:.6f}")

    # Сохраняем лучшую модель
    model_map = {
        'Настроенный RandomForest': best_rf_model,
        'Настроенный LightGBM': best_lgbm_model,
        'Стекинг (RF+LGBM)': stacking_regressor
    }
    final_model_to_save = model_map.get(best_model_name, best_lgbm_model)
    model_filename = 'final_best_model.joblib'
    print(f"\n--- Сохранение лучшей модели '{best_model_name}' в файл '{model_filename}' ---")
    joblib.dump(final_model_to_save, model_filename)
    print("✅ Модель успешно сохранена.")

    # Пример загрузки и использования
    print("\n--- Проверка загрузки и использования модели ---")
    loaded_model = joblib.load(model_filename)
    print("✅ Модель загружена.")
    sample_to_predict = X_test.iloc[[0]]
    prediction = loaded_model.predict(sample_to_predict)
    print(f"Предсказание для первого объекта из тестовой выборки: {prediction[0]:.4f}")
    print(f"Реальное значение: {y_test.iloc[0]:.4f}")


if __name__ == "__main__":
    main()