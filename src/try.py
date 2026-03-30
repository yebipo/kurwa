# =========================================================================
# ===    СКРИПТ ДЛЯ ОБУЧЕНИЯ, НАСТРОЙКИ И СРАВНЕНИЯ МОДЕЛЕЙ (V3)        ===
# =========================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
import lightgbm as lgb
import time
import warnings

# Игнорировать предупреждения для более чистого вывода
warnings.filterwarnings('ignore')

# --- 1. КОНФИГУРАЦИЯ ---
# Измените эти параметры под вашу задачу
# -------------------------------------------------------------------------
# Используйте имя вашего CSV-файла
FILE_PATH = 'output_adequate.csv'

# !!! ВАЖНО: Используем числовые индексы, так как в файле нет заголовков.
FEATURE_COLUMNS = [1, 2, 3, 4]  # Индексы колонок-признаков (X)
TARGET_COLUMN = 5  # Индекс целевой колонки (y)

TEST_SIZE = 0.2  # Доля данных для тестовой выборки (20%)
RANDOM_STATE = 42  # Фиксированное значение для воспроизводимости результатов

# Расширенные параметры для поиска по сетке (GridSearchCV) для RandomForest
RF_PARAM_GRID_EXTENDED = {
    'n_estimators': [200, 400],
    'max_depth': [30, None],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt', 0.7]
}

# Параметры для поиска по сетке для LightGBM
LGBM_PARAM_GRID = {
    'n_estimators': [400, 700, 1000],
    'learning_rate': [0.05, 0.1],
    'num_leaves': [31, 50],
}


# -------------------------------------------------------------------------


def load_and_prepare_data(file_path, feature_cols, target_col, test_size, random_state):
    """
    Загружает данные из CSV, ПРОПУСКАЯ ПЕРВУЮ СТРОКУ, удаляет NaN и разделяет.
    """
    print("--- ЭТАП 1: Загрузка и подготовка данных ---")
    try:
        all_required_cols = feature_cols + [target_col]
        df = pd.read_csv(
            file_path,
            decimal=',',
            sep=';',
            header=None,
            usecols=all_required_cols,
            skiprows=1  # <--- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: Пропускаем первую строку файла
        )
        print("✅ Данные успешно загружены, первая строка пропущена.")
        print(f"   Анализируется {df.shape[0]} строк и {df.shape[1]} колонок.")
    except FileNotFoundError:
        print(f"❌ ОШИБКА: Файл не найден по пути '{file_path}'.")
        return None
    except ValueError as e:
        print(f"❌ ПРОИЗОШЛА ОШИБКА ПРИ ЧТЕНИИ ФАЙЛА: {e}")
        print("   Возможная причина: в файле отсутствуют колонки, указанные в конфигурации.")
        return None
    except Exception as e:
        print(f"❌ ПРОИЗОШЛА ОШИБКА ПРИ ЧТЕНИИ ФАЙЛА: {e}")
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
    """Универсальная функция для выполнения GridSearchCV."""
    print(f"--- Поиск лучших параметров для {model_name} (GridSearchCV) ---")

    try:
        estimator.set_params(random_state=random_state)
    except ValueError:
        pass

    grid_search = GridSearchCV(
        estimator=estimator, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2'
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

    # --- ЭТАП 2: Обучение базовой модели для контроля ---
    print("\n--- ЭТАП 2: Обучение базовой модели RandomForest ---")
    base_rf = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    base_rf.fit(X_train, y_train)
    base_score = base_rf.score(X_test, y_test)
    print(f"   Базовый R-squared (n_estimators=100): {base_score:.6f}\n")
    results['Базовый RandomForest'] = {'R-squared': base_score, 'Время обучения (сек)': '-'}

    # --- ЭТАП 3: Настройка продвинутых моделей ---
    print("\n--- ЭТАП 3: Настройка и сравнение продвинутых моделей ---")

    # RandomForest с расширенным поиском
    best_rf_model = run_grid_search(
        X_train, y_train, RandomForestRegressor(), RF_PARAM_GRID_EXTENDED, "RandomForest", RANDOM_STATE
    )
    score_gs_rf = best_rf_model.score(X_test, y_test)
    results['Настроенный RandomForest'] = {'R-squared': score_gs_rf, 'Время обучения (сек)': '-'}

    # LightGBM с поиском по сетке
    best_lgbm_model = run_grid_search(
        X_train, y_train, lgb.LGBMRegressor(), LGBM_PARAM_GRID, "LightGBM", RANDOM_STATE
    )
    score_gs_lgbm = best_lgbm_model.score(X_test, y_test)
    results['Настроенный LightGBM'] = {'R-squared': score_gs_lgbm, 'Время обучения (сек)': '-'}

    # --- ЭТАП 4: Инжиниринг признаков (PolynomialFeatures) ---
    print("\n--- ЭТАП 4: Эксперимент с инжинирингом признаков ---")

    poly_pipeline = Pipeline([
        ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('regressor', best_rf_model)  # Используем уже найденную лучшую модель RF
    ])

    print("--- Обучение конвейера (PolynomialFeatures + StandardScaler + RandomForest) ---")
    start_time = time.time()
    poly_pipeline.fit(X_train, y_train)
    training_time = time.time() - start_time

    poly_score = poly_pipeline.score(X_test, y_test)
    print(f"   Обучение завершено за {training_time:.2f} сек.")
    print(f"   Качество на тесте с новыми признаками (R-squared): {poly_score:.6f}\n")
    results['RF + PolyFeatures'] = {'R-squared': poly_score, 'Время обучения (сек)': training_time}

    # --- ЭТАП 5: Итоговое сравнение ---
    print("==================================================")
    print("###          ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ        ###")
    print("==================================================")

    results_df = pd.DataFrame(results).T.sort_values(by='R-squared', ascending=False)
    # Форматирование для красивого вывода
    formatters = {
        'R-squared': '{:,.6f}'.format,
        'Время обучения (сек)': lambda x: f"{x:,.2f}" if pd.notna(x) and x != '-' else '-'
    }
    print(results_df.to_string(formatters=formatters))

    best_model_name = results_df.index[0]
    best_score = results_df['R-squared'].iloc[0]
    print(f"\n🏆🏆🏆 Лучший подход: '{best_model_name}' с R-squared = {best_score:.6f}")


if __name__ == "__main__":
    main()