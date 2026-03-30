import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
# --- НОВЫЙ БЛОК: Импорт метрик ---
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# --- 1. КОНФИГУРАЦИЯ ---
file_path = 'output_run_2025_09_24_12_22_07.xlsx'

try:
    # --- 2. ЗАГРУЗКА И ОБУЧЕНИЕ МОДЕЛИ ---
    print("Загрузка и подготовка данных...")
    # Загружаем из Excel, используя первую строку как заголовок
    data_full = pd.read_excel(file_path, sheet_name=0, header=0)

    # Выбираем нужные колонки и удаляем строки с пропусками
    df = data_full.iloc[:, 1:6].copy()
    df.dropna(inplace=True)

    # Определяем признаки и цель по именам, сгенерированным pandas
    X = df[['Unnamed: 1', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4']]
    y = df['Unnamed: 5']

    # Разделяем данные
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Обучение модели Случайного Леса...")
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    print("Обучение завершено.\n")

    # --- 3. ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ ---
    print("Вычисление предсказаний...")
    y_pred = rf_model.predict(X_test)

    # --- НОВЫЙ БЛОК: РАСЧЕТ И ВЫВОД ВСЕХ МЕТРИК ---
    print("=" * 50)
    print("### Метрики Качества Базовой Модели ###")
    print("=" * 50)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # Аккуратный расчет MAPE, чтобы избежать деления на ноль
    y_test_for_mape = y_test.copy()
    # Заменяем нули на очень маленькое число, если они есть
    y_test_for_mape[y_test_for_mape == 0] = 1e-9
    mape = np.mean(np.abs((y_test - y_pred) / y_test_for_mape)) * 100

    print(f"  R-squared (Коэффициент детерминации): {r2:.6f}")
    print(f"  MAE (Средняя абсолютная ошибка):        {mae:.6f}")
    print(f"  RMSE (Корень из среднеквадратичной ошибки): {rmse:.6f}")
    print(f"  MAPE (Средняя абсолютная процентная ошибка):  {mape:.2f}%\n")
    # --- КОНЕЦ НОВОГО БЛОКА ---

    # --- 4. ПОСТРОЕНИЕ ГРАФИКОВ ОШИБОК ---
    print("=" * 50)
    print("### Анализ Погрешностей Модели ###")
    print("=" * 50)
    print("Строятся графики... Пожалуйста, посмотрите на появившееся окно.")

    # Вычисляем погрешности (остатки)
    residuals = y_test - y_pred

    # Создаем фигуру с двумя графиками рядом
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Анализ погрешностей предсказаний базовой модели', fontsize=18)

    # --- График 1: Предсказания vs. Реальные значения ---
    axes[0].scatter(y_test, y_pred, alpha=0.3, s=15, edgecolors='k', linewidths=0.5)
    lims = [
        np.min([axes[0].get_xlim(), axes[0].get_ylim()]),
        np.max([axes[0].get_xlim(), axes[0].get_ylim()]),
    ]
    axes[0].plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Идеальное предсказание')
    axes[0].set_xlabel('Реальные значения', fontsize=12)
    axes[0].set_ylabel('Предсказанные значения', fontsize=12)
    axes[0].set_title('Предсказания vs. Реальные значения', fontsize=14)
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].grid(True)
    axes[0].legend()

    # --- График 2: Погрешности vs. Предсказания ---
    axes[1].scatter(y_pred, residuals, alpha=0.3, s=15, edgecolors='k', linewidths=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', label='Нулевая погрешность')
    axes[1].set_xlabel('Предсказанные значения', fontsize=12)
    axes[1].set_ylabel('Погрешность (Реальное - Предсказанное)', fontsize=12)
    axes[1].set_title('График погрешностей (Residual Plot)', fontsize=14)
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

except Exception as e:
    print(f"Произошла непредвиденная ошибка: {e}")