import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# --- 1. КОНФИГУРАЦИЯ ---
file_path = 'output_run_2025_09_24_12_22_07.xlsx'

try:
    # --- 2. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ---
    print("Загрузка и подготовка данных...")
    data_full = pd.read_excel(file_path, sheet_name=0, header=0)

    # Берем исходные колонки: 1-4 как признаки, 5 как цель
    df = data_full.iloc[:, 1:6].copy()
    df.dropna(inplace=True)
    print(f"Анализируется {len(df)} строк.\n")

    # Определяем наши "сырые" признаки (X) и цель (Y)
    # Используем исходные колонки, а не произведения!
    X = df[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']]
    y = df['Unnamed: 5']

    # --- 3. РАЗДЕЛЕНИЕ ДАННЫХ НА ОБУЧАЮЩУЮ И ТЕСТОВУЮ ВЫБОРКИ ---
    # Это стандартная практика в машинном обучении, чтобы честно оценить качество модели.
    # Модель учится на 80% данных, а проверяется на 20% данных, которые она не видела.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Данные разделены на обучающую ({len(X_train)} строк) и тестовую ({len(X_test)} строк) выборки.\n")

    # --- 4. СОЗДАНИЕ И ОБУЧЕНИЕ МОДЕЛИ СЛУЧАЙНОГО ЛЕСА ---
    print("Обучение модели Случайного Леса... Это может занять некоторое время.")

    # n_estimators=100 - создаем "лес" из 100 деревьев.
    # random_state=42 - для воспроизводимости результатов.
    # n_jobs=-1 - использовать все доступные ядра процессора для ускорения.
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

    # Обучаем модель на тренировочных данных
    rf_model.fit(X_train, y_train)
    print("Обучение завершено.\n")

    # --- 5. ОЦЕНКА КАЧЕСТВА МОДЕЛИ ---
    print("=" * 50)
    print("### 1. Качество Модели (R-squared) ###")
    print("=" * 50)

    # Делаем предсказания на тестовых данных, которые модель не видела
    y_pred = rf_model.predict(X_test)

    # Считаем R-squared
    r2 = r2_score(y_test, y_pred)
    print(f"R-squared на тестовых данных: {r2:.4f}")
    if r2 < 0.2:
        print("Качество модели невысокое. Возможно, в данных не хватает важных признаков.")
    elif r2 < 0.7:
        print("Качество модели среднее. Зависимость есть, но она сложная или зашумленная.")
    else:
        print("Отличное качество модели! Большинство зависимостей найдено.")

    # --- 6. АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ ---
    print("\n" + "=" * 50)
    print("### 2. Важность Признаков ###")
    print("=" * 50)

    importances = rf_model.feature_importances_
    feature_names = X.columns

    # Создаем DataFrame для удобного отображения
    importance_df = pd.DataFrame({'Признак': feature_names, 'Важность': importances})
    importance_df = importance_df.sort_values(by='Важность', ascending=False)

    print("Оценка вклада каждого признака в предсказание энтропии:")
    print(importance_df.to_string(index=False))

    # Визуализация важности признаков
    plt.figure(figsize=(10, 6))
    plt.bar(importance_df['Признак'], importance_df['Важность'], color='skyblue')
    plt.title('Важность признаков по оценке Случайного Леса', fontsize=16)
    plt.ylabel('Важность', fontsize=12)
    plt.xlabel('Признаки', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")