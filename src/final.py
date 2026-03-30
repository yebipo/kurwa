import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import lightgbm as lgb
import time

# --- КОД, КОТОРЫЙ ДОЛЖЕН БЫТЬ СНАРУЖИ ---
# Константы и импорты можно оставить здесь
file_path = 'output_run_2025_09_24_12_22_07.xlsx'

# --- НАЧАЛО ЗАЩИТНОГО БЛОКА ---
if __name__ == '__main__':
    try:
        # --- 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---
        print("Загрузка и подготовка данных...")
        data_full = pd.read_excel(file_path, sheet_name=0, header=0)

        df = data_full.iloc[:, 1:6].copy()
        df.dropna(inplace=True)
        print(f"Анализируется {len(df)} строк.\n")

        X = df[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']]
        y = df['Unnamed: 5']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"Данные разделены на обучающую ({len(X_train)}) и тестовую ({len(X_test)}) выборки.\n")

        # =========================================================================
        # === БАЗОВАЯ МОДЕЛЬ: Случайный Лес по умолчанию (для сравнения) ===
        # =========================================================================
        print("--- Обучение базовой модели RandomForest (для сравнения) ---")
        start_time = time.time()
        base_rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        base_rf.fit(X_train, y_train)
        base_pred = base_rf.predict(X_test)
        base_r2 = r2_score(y_test, base_pred)
        print(f"Обучение завершено за {time.time() - start_time:.2f} секунд.\n")

        # =========================================================================
        # === ЭТАП 1: Тонкая настройка RandomForest с GridSearchCV ===
        # =========================================================================
        print("--- ЭТАП 1: Запуск GridSearchCV для RandomForest ---")
        print("Это может занять ОЧЕНЬ много времени. verbose=2 будет показывать прогресс.")

        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [20, 30, None],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 1.0]
        }

        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='r2')

        start_time = time.time()
        grid_search.fit(X_train, y_train)
        print(f"Поиск по сетке завершен за {time.time() - start_time:.2f} секунд.\n")

        print("Лучшие найденные параметры для RandomForest:")
        print(grid_search.best_params_)

        best_rf_model = grid_search.best_estimator_
        tuned_pred = best_rf_model.predict(X_test)
        tuned_r2 = r2_score(y_test, tuned_pred)

        # =========================================================================
        # === ЭТАП 2: Обучение модели LightGBM ===
        # =========================================================================
        print("\n--- ЭТАП 2: Обучение модели LightGBM ---")
        start_time = time.time()
        lgb_model = lgb.LGBMRegressor(random_state=42, n_jobs=-1)
        lgb_model.fit(X_train, y_train)
        lgb_pred = lgb_model.predict(X_test)
        lgb_r2 = r2_score(y_test, lgb_pred)
        print(f"Обучение завершено за {time.time() - start_time:.2f} секунд.\n")

        # =========================================================================
        # === ИТОГОВОЕ СРАВНЕНИЕ ===
        # =========================================================================
        print("\n" + "=" * 50)
        print("### ИТОГОВОЕ СРАВНЕНИЕ МОДЕЛЕЙ ###")
        print("=" * 50)
        print(f"1. Базовый RandomForest:           R-squared = {base_r2:.6f}")
        print(f"2. Настроенный RandomForest (GridSearchCV): R-squared = {tuned_r2:.6f}")
        print(f"3. LightGBM:                      R-squared = {lgb_r2:.6f}")

        results = {
            "Базовый RandomForest": base_r2,
            "Настроенный RandomForest": tuned_r2,
            "LightGBM": lgb_r2
        }
        best_model_name = max(results, key=results.get)
        print(f"\n🏆 Лучшая модель: {best_model_name} с R-squared = {results[best_model_name]:.6f}")

    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")