# ===================================================================================
# ===   СКРИПТ ДЛЯ АНАЛИЗА ГОТОВОЙ "ЧЕМПИОНСКОЙ" МОДЕЛИ С ПОМОЩЬЮ SHAP      ===
# ===        (V2.6 - Скрипт сделан универсальным для любого количества признаков)       ===
# ===================================================================================

import pandas as pd
import joblib
import shap
from sklearn.model_selection import train_test_split
import warnings
import time

# Импорты нужны, чтобы pickle мог восстановить классы вашей модели
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')


# Класс-заглушка для успешной загрузки объекта FinalModel, если он был сохранен внутри контейнера
class FinalModel:
    pass


# --- 1. КОНФИГУРАЦИЯ ---
MODEL_PATH = 'model_log_2par.joblib'  # Путь к вашей сохраненной модели
FULL_DATA_PATH = 'data_for_shap_analysis.csv'  # Файл с признаками, который использовался для обучения

ANALYSIS_DATA_RATIO = 0.3  # Доля данных для анализа (чтобы не ждать вечно)
RANDOM_STATE = 42

print("--- Запуск скрипта для интерпретации модели (V2.6) ---")
start_time = time.time()

# --- 2. ЗАГРУЗКА МОДЕЛИ И ДАННЫХ ---
try:
    print(f"\n[Шаг 1/5] Загрузка модели-контейнера из '{MODEL_PATH}'...")
    model_container = joblib.load(MODEL_PATH)
    actual_model = model_container.stacking_model
    print("   ✅ Успешно извлечена реальная модель ('stacking_model') из контейнера!")

    print(f"\n[Шаг 2/5] Загрузка полного набора данных из '{FULL_DATA_PATH}'...")
    # Читаем CSV с разделителем-запятой
    full_dataset = pd.read_csv(FULL_DATA_PATH)

    # Теперь скрипт автоматически определяет признаки из файла
    features_list = full_dataset.columns.tolist()
    print(f"   ✅ Данные загружены. Всего строк: {len(full_dataset)}.")
    print(f"   ✅ Обнаружены следующие признаки для анализа: {features_list}")

    # =============================================================================
    # === ИЗМЕНЕНИЕ: Жесткая проверка на 5 колонок убрана.                     ===
    # === Теперь скрипт будет работать с любым количеством признаков в файле.    ===
    # =============================================================================
    # if len(full_dataset.columns) != 5:
    #     print(f"\n❌ ОШИБКА: Ожидалось 5 колонок в файле, но найдено {len(full_dataset.columns)}.")
    #     exit()

except Exception as e:
    import traceback
    print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА на этапе загрузки: {e}")
    traceback.print_exc()
    exit()

# --- 3. РАЗДЕЛЕНИЕ ДАННЫХ ДЛЯ ЦЕЛЕЙ АНАЛИЗА ---
# Мы не будем анализировать весь датасет, чтобы сэкономить время.
# Возьмем только часть данных для расчета SHAP.
print(f"\n[Шаг 3/5] Разделение данных для SHAP-анализа...")
X_background_full, X_for_analysis = train_test_split(
    full_dataset, test_size=ANALYSIS_DATA_RATIO, random_state=RANDOM_STATE
)
print(f"   ✅ Данные для анализа готовы ({len(X_for_analysis)} строк).")

# --- 4. РАСЧЕТ SHAP-ЗНАЧЕНИЙ ---
print("\n" + "=" * 60)
print("###          АНАЛИЗ МОДЕЛИ С ПОМОЩЬЮ SHAP           ###")
print("=" * 60)

# KernelExplainer требует "фоновый" набор данных для расчета средних значений.
# Чтобы ускорить процесс, берем небольшую выборку (100-200 строк).
X_background_sample = shap.sample(X_background_full, 100, random_state=RANDOM_STATE)

print(f"\n[Шаг 4/5] Создание объяснителя и расчет SHAP-значений...")
print("   (Это самый долгий этап, он может занять несколько минут. Пожалуйста, подождите...)")

# Используем KernelExplainer, так как он универсален и подходит для сложных моделей вроде StackingRegressor.
explainer = shap.KernelExplainer(actual_model.predict, X_background_sample)
shap_values = explainer.shap_values(X_for_analysis)
print("\n   ✅ Расчет SHAP-значений завершен!")

# --- 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ---
print(f"\n[Шаг 5/5] Генерация и отображение графиков анализа...")

try:
    print("\n--> Отображается [Анализ 1]: Глобальная важность признаков (bar plot)...")
    shap.summary_plot(shap_values, X_for_analysis, plot_type="bar", show=True)
    print("    Закройте окно с графиком, чтобы продолжить...")

    print("\n--> Отображается [Анализ 2]: Карта влияния признаков (beeswarm plot)...")
    shap.summary_plot(shap_values, X_for_analysis, show=True)
    print("    Закройте окно с графиком, чтобы продолжить...")

    i = 0  # Берем первый объект из выборки для детального анализа
    print(f"\n--> Отображается [Анализ 3]: Разбор прогноза для одного объекта (индекс {i})...")
    # matplotlib=True нужен, чтобы график можно было сохранить или показать в средах, где нет JS.
    shap.force_plot(explainer.expected_value, shap_values[i, :], X_for_analysis.iloc[i, :], matplotlib=True, show=True)
    print("    Закройте окно с графиком, чтобы продолжить...")

    # Для детального графика зависимости выберем последний признак (скорее всего, это 'cluster')
    # Вы можете заменить его на любую другую колонку, например: interesting_feature = 'param_1'
    interesting_feature = X_for_analysis.columns[-1]
    print(f"\n--> Отображается [Анализ 4]: График зависимости для признака '{interesting_feature}'...")
    shap.dependence_plot(interesting_feature, shap_values, X_for_analysis, interaction_index=None, show=True)
    print("    Закройте окно с графиком для завершения.")

except Exception as e:
    print(f"\n❌ ОШИБКА при генерации графиков: {e}")
    print("    Возможно, одно из окон было закрыто досрочно. Расчеты SHAP завершены, но визуализация прервана.")

finally:
    end_time = time.time()
    print(f"\n--- Анализ полностью завершен за {end_time - start_time:.2f} секунд ---")