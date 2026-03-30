# =========================================================================================
# ===       СКРИПТ ДЛЯ АНАЛИЗА МУЛЬТИКОЛЛИНЕАРНОСТИ И СРАВНЕНИЯ ЛИНЕЙНЫХ МОДЕЛЕЙ       ===
# =========================================================================================

import pandas as pd
import numpy as np
import time
import warnings

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# Модели и утилиты
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

# Модели для анализа мультиколлинеарности
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

# Игнорировать предупреждения для более чистого вывода
warnings.filterwarnings('ignore')

# --- 1. КОНФИГУРАЦИЯ ---
# -------------------------------------------------------------------------
FILE_PATH = 'output_adequate.csv'

# !!! Используем числовые индексы, так как в файле нет заголовков.
# Предположим, что X1, X2 - это колонки 1, 2, а X3, X4 - это 3, 4
FEATURE_COLUMNS = [1, 2, 3, 4]  # Индексы колонок-признаков (X)
TARGET_COLUMN = 5  # Индекс целевой колонки (y)

# Создадим имена для признаков для удобства анализа
FEATURE_NAMES = {1: 'X1', 2: 'X2', 3: 'X3', 4: 'X4'}

TEST_SIZE = 0.2
RANDOM_STATE = 42


# -------------------------------------------------------------------------

def load_and_prepare_data(file_path, feature_cols, target_col, test_size, random_state):
    """
    Загружает данные, переименовывает колонки, удаляет NaN и разделяет.
    """
    print("--- ЭТАП 0: Загрузка и подготовка данных ---")
    try:
        all_required_cols = feature_cols + [target_col]
        df = pd.read_csv(
            file_path, decimal=',', sep=';', header=None,
            usecols=all_required_cols, skiprows=1
        )
        df.rename(columns=FEATURE_NAMES, inplace=True)
        if target_col not in FEATURE_NAMES:
            df.rename(columns={target_col: 'target'}, inplace=True)
        else:
            df['target'] = df[FEATURE_NAMES[target_col]]

        print("✅ Данные успешно загружены и колонки переименованы.")
    except Exception as e:
        print(f"❌ ПРОИЗОШЛА ОШИБКА ПРИ ЧТЕНИИ ФАЙЛА: {e}")
        return None

    # Очистка и разделение
    initial_rows = len(df)
    target_col_name = 'target'
    df[target_col_name] = pd.to_numeric(df[target_col_name], errors='coerce')
    df.dropna(subset=[target_col_name], inplace=True)
    if len(df) < initial_rows:
        print(f"⚠️  Удалено {initial_rows - len(df)} строк с нечисловыми/пустыми значениями в цели.")

    feature_names_list = list(FEATURE_NAMES.values())
    X = df[feature_names_list]
    y = df[target_col_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Данные разделены на обучающую ({len(X_train)}) и тестовую ({len(X_test)}) выборки.\n")
    return X_train, X_test, y_train, y_test


def run_diagnostics(X_train, feature_names):
    """
    Проводит диагностику мультиколлинеарности: корреляционная матрица и VIF.
    """
    print("--- ЭТАП 1: Диагностика мультиколлинеарности ---")

    # 1. Корреляционная матрица
    print("   1. Корреляционная матрица:")
    corr_matrix = X_train.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Матрица корреляций признаков')
    plt.show()
    print("      (см. график)")

    # 2. VIF (Variance Inflation Factor)
    print("\n   2. Фактор инфляции дисперсии (VIF):")
    X_const = add_constant(X_train.values)
    vif_data = pd.DataFrame()
    vif_data["feature"] = feature_names
    vif_data["VIF"] = [variance_inflation_factor(X_const, i + 1) for i in range(len(feature_names))]

    print(vif_data)
    high_vif = vif_data[vif_data['VIF'] > 10]
    if not high_vif.empty:
        print("\n   🚨 ВЫВОД: Обнаружена сильная мультиколлинеарность (VIF > 10) для признаков:",
              list(high_vif['feature']))
    else:
        print("\n   ✅ ВЫВОД: Сильная мультиколлинеарность не обнаружена (VIF < 10).")
    print("-" * 50 + "\n")


def main():
    """Главная функция, запускающая весь процесс."""
    start_total_time = time.time()

    prepared_data = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    if prepared_data is None: return
    X_train, X_test, y_train, y_test = prepared_data

    run_diagnostics(X_train, list(FEATURE_NAMES.values()))

    results = {}
    coefficients = {}

    # --- ЭТАП 2: Сравнение моделей для борьбы с мультиколлинеарностью ---
    print("\n--- ЭТАП 2: Обучение и сравнение моделей ---")

    # 2.1. Линейная регрессия (полный набор) - для сравнения
    print("\n--- 2.1 Базовая линейная регрессия (на всех признаках) ---")
    base_lr_pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())])
    base_lr_pipeline.fit(X_train, y_train)
    score_base_lr = base_lr_pipeline.score(X_test, y_test)
    results['LR (Все признаки)'] = {'R-squared': score_base_lr}
    coefficients['LR (Все признаки)'] = base_lr_pipeline.named_steps['regressor'].coef_
    print(f"   Качество на тесте (R²): {score_base_lr:.6f} (Эта модель может страдать от мультиколлинеарности)\n")

    # 2.2. Линейная регрессия с исключением признаков
    print("--- 2.2 Линейная регрессия с исключением признаков (только X1, X2) ---")
    X_train_reduced = X_train[['X1', 'X2']]
    X_test_reduced = X_test[['X1', 'X2']]

    lr_pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', LinearRegression())])
    lr_pipeline.fit(X_train_reduced, y_train)
    score_lr = lr_pipeline.score(X_test_reduced, y_test)
    results['LR (только X1, X2)'] = {'R-squared': score_lr}
    # Сохраняем коэффициенты для X1, X2 и ставим 0 для остальных
    coefs_reduced = lr_pipeline.named_steps['regressor'].coef_
    coefficients['LR (только X1, X2)'] = [coefs_reduced[0], coefs_reduced[1], 0, 0]
    print(f"   Качество на тесте (R²): {score_lr:.6f}")
    print(f"   Стабильные коэффициенты: X1={coefs_reduced[0]:.3f}, X2={coefs_reduced[1]:.3f}\n")

    # 2.3. PCA Регрессия
    print("--- 2.3 Регрессия на главные компоненты (PCA) ---")
    pca_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=2, random_state=RANDOM_STATE)),
        ('regressor', LinearRegression())
    ])
    pca_pipeline.fit(X_train, y_train)
    score_pca = pca_pipeline.score(X_test, y_test)
    results['LR + PCA (2 комп.)'] = {'R-squared': score_pca}
    # Коэффициенты для PCA-регрессии неинтерпретируемы в терминах исходных признаков
    # === ИСПРАВЛЕНИЕ ЗДЕСЬ ===
    coefficients['LR + PCA (2 комп.)'] = [np.nan] * 4
    print(f"   Качество на тесте (R²): {score_pca:.6f}\n")

    # 2.4. Ridge-регрессия (L2 регуляризация)
    print("--- 2.4 Ridge-регрессия (с подбором alpha) ---")
    ridge_pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', Ridge(random_state=RANDOM_STATE))])
    ridge_param_grid = {'regressor__alpha': np.logspace(-2, 3, 10)}  # Более широкий поиск alpha
    grid_ridge = GridSearchCV(ridge_pipeline, ridge_param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_ridge.fit(X_train, y_train)
    score_ridge = grid_ridge.score(X_test, y_test)
    results['Ridge-регрессия'] = {'R-squared': score_ridge}
    coefficients['Ridge-регрессия'] = grid_ridge.best_estimator_.named_steps['regressor'].coef_
    print(f"   Качество на тесте (R²): {score_ridge:.6f}")
    print(f"   Лучший параметр alpha: {grid_ridge.best_params_['regressor__alpha']:.4f}\n")

    # 2.5. Lasso-регрессия (L1 регуляризация)
    print("--- 2.5 Lasso-регрессия (с подбором alpha) ---")
    lasso_pipeline = Pipeline([('scaler', StandardScaler()), ('regressor', Lasso(random_state=RANDOM_STATE))])
    lasso_param_grid = {'regressor__alpha': np.logspace(-4, 0, 10)}  # У Lasso alpha обычно меньше
    grid_lasso = GridSearchCV(lasso_pipeline, lasso_param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_lasso.fit(X_train, y_train)
    score_lasso = grid_lasso.score(X_test, y_test)
    results['Lasso-регрессия'] = {'R-squared': score_lasso}
    coefficients['Lasso-регрессия'] = grid_lasso.best_estimator_.named_steps['regressor'].coef_
    print(f"   Качество на тесте (R²): {score_lasso:.6f}")
    print(f"   Лучший параметр alpha: {grid_lasso.best_params_['regressor__alpha']:.4f}\n")

    # --- ЭТАП 3: Итоговое сравнение ---
    print("\n" + "=" * 80)
    print("###                    ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ                    ###")
    print("=" * 80)

    # Таблица с R-квадрат
    results_df = pd.DataFrame(results).T.sort_values(by='R-squared', ascending=False)
    results_df['R-squared'] = results_df['R-squared'].apply('{:,.6f}'.format)
    print("\n--- Таблица 1: Сравнение качества моделей (R-squared) ---")
    print(results_df)

    # Таблица с коэффициентами
    coef_df = pd.DataFrame(coefficients, index=list(FEATURE_NAMES.values())).T
    # Переупорядочим для соответствия таблице результатов
    coef_df = coef_df.loc[results_df.index]

    print("\n\n--- Таблица 2: Сравнение коэффициентов моделей ---")
    print("(Коэффициенты показаны для масштабированных данных)")
    print(coef_df.to_string(float_format="%.4f"))

    print("\n\n--- Анализ результатов ---")
    print("1. Сравните R-squared в Таблице 1. Какая модель дает лучший прогноз?")
    print(
        "2. Посмотрите на 'LR (Все признаки)' в Таблице 2. Если VIF высокий, коэффициенты могут быть огромными, с разными знаками.")
    print("3. 'LR (только X1, X2)' дает стабильные, интерпретируемые коэффициенты для базовой модели.")
    print("4. 'Ridge-регрессия' 'ужимает' все коэффициенты, делая их более стабильными, но не обнуляет.")
    print(
        "5. 'Lasso-регрессия', скорее всего, обнулила коэффициенты у избыточных признаков, автоматически выполнив их отбор.")

    total_time = time.time() - start_total_time
    print(f"\nОбщее время выполнения скрипта: {total_time:.2f} секунд.")


if __name__ == "__main__":
    main()