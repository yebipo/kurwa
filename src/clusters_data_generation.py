# =================================================================================
# ===   СКРИПТ ДЛЯ ГЕНЕРАЦИИ 5-КОЛОНОЧНЫХ ДАННЫХ ИЗ ГОТОВОЙ МОДЕЛИ           ===
# =================================================================================
#
#   Цель: Загрузить обученную модель и сырые данные, применить к данным
#         внутреннюю логику предобработки модели (масштабирование + кластеризацию)
#         и сохранить результат (данные с 5 признаками) в новый CSV-файл.
#

import pandas as pd
import joblib
import warnings
# Импорты нужны, чтобы pickle мог восстановить классы
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

# Класс-заглушка для успешной загрузки объекта FinalModel
class FinalModel:
    pass

# --- 1. КОНФИГУРАЦИЯ ---
# ВХОДНЫЕ ФАЙЛЫ
MODEL_PATH = 'model.joblib' # Ваша сохраненная модель
RAW_DATA_PATH = 'output_adequate.csv'      # Ваши сырые данные

# ВЫХОДНОЙ ФАЙЛ
OUTPUT_FILE_PATH = 'data_with_clusters.csv' # Имя файла, куда мы сохраним результат

# Параметры, соответствующие вашему скрипту обучения
FEATURE_COLUMNS = [1, 2, 3, 4]


def main():
    """
    Главная функция для генерации признаков.
    """
    print("--- Запуск скрипта для генерации признаков ---")

    # --- Шаг 1: Загрузка артефактов ---
    try:
        print(f"\n[Шаг 1/3] Загрузка обученной модели-контейнера из '{MODEL_PATH}'...")
        # loaded_model - это ваш объект FinalModel со всеми обученными компонентами
        loaded_model = joblib.load(MODEL_PATH)
        print("   ✅ Модель успешно загружена.")

        print(f"\n[Шаг 2/3] Загрузка сырых 4-колоночных данных из '{RAW_DATA_PATH}'...")
        # Загружаем только 4 колонки с признаками
        raw_data = pd.read_csv(
            RAW_DATA_PATH, decimal=',', sep=';', header=None,
            usecols=FEATURE_COLUMNS, skiprows=1
        )
        raw_data.columns = [f'feature_{i}' for i in range(1, 5)] # Даем нормальные имена
        print(f"   ✅ Загружено {len(raw_data)} строк с 4 признаками.")

    except Exception as e:
        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА при загрузке: {e}")
        return

    # --- Шаг 2: Генерация признака кластера ---
    print("\n[Шаг 3/3] Генерация признака 'cluster_id'...")

    # Ваш класс FinalModel очень удобен. Он хранит обученные scaler и kmeans.
    # Мы просто вызовем внутреннюю логику, которую вы написали.
    # Это гарантирует 100% идентичную обработку данных.

    # 1. Применяем обученный scaler
    X_scaled = loaded_model.scaler.transform(raw_data)
    print("   - Масштабирование применено.")

    # 2. Применяем обученный kmeans для получения меток кластеров
    cluster_labels = loaded_model.kmeans.predict(X_scaled)
    print("   - Кластеры предсказаны.")

    # 3. Собираем новый DataFrame с 5 признаками
    data_with_clusters = raw_data.copy()
    data_with_clusters['cluster_id'] = cluster_labels
    print("   - Финальный DataFrame с 5 признаками собран.")

    # --- Шаг 3: Сохранение результата ---
    try:
        print(f"\n--- Сохранение результата в файл '{OUTPUT_FILE_PATH}' ---")
        data_with_clusters.to_csv(OUTPUT_FILE_PATH, index=False)
        print(f"\n✅ УСПЕХ! Создан файл '{OUTPUT_FILE_PATH}' с 5 признаками.")
        print("   Теперь вы можете использовать этот файл как 'full_data.csv' в скрипте для SHAP-анализа.")
    except Exception as e:
        print(f"\n❌ ОШИБКА при сохранении файла: {e}")


if __name__ == "__main__":
    main()