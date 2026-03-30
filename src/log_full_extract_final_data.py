# =========================================================================================
# === ГЕНЕРАТОР ОТЧЕТА (МЕТРИКИ + ПАРАМЕТРЫ) ===
# === Воссоздает точный формат отчета без долгого поиска Optuna ===
# =========================================================================================
import pandas as pd
import numpy as np
import optuna
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from yellowbrick.cluster import KElbowVisualizer
import warnings

# --- НАСТРОЙКИ (Проверьте имя вашей базы данных!) ---
OPTUNA_DB_NAME = "optuna_v26.db"  # <--- ИМЯ ВАШЕГО ФАЙЛА С БАЗОЙ (v25 или v26)
OPTUNA_STUDY_NAME = "stacking_v26_boosted"  # Имя исследования (если v25, то stacking_v25)

# Если не помните имя study, скрипт попробует угадать или вывести список, см. ниже.
# Для v25 study_name="stacking_v25", для v26="stacking_v26_boosted"

FILE_PATH = 'output_adequate.csv'
REPORT_FILENAME = "FINAL_REPORT_FULL.txt"
RANDOM_STATE = 42
TEST_SIZE_SHAP = 0.20
MAX_CLUSTERS_TO_TEST = 7

COLUMN_MAPPING = {1: 'feature_1', 2: 'feature_2', 3: 'feature_3', 4: 'feature_4', 5: 'target'}
FEATURE_COLUMNS = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
TARGET_COLUMN = 'target'

warnings.filterwarnings('ignore')


# --- КЛАССЫ (Копия логики из основного скрипта) ---
class GlobalPreprocessor:
    def __init__(self, scaler, kmeans):
        self.scaler = scaler
        self.kmeans = kmeans

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        cluster_labels = self.kmeans.predict(X_scaled)
        X_out = X_log.copy()
        X_out['cluster_id'] = cluster_labels
        return X_out


class ProductionModel:
    def __init__(self, preprocessor, lgbm_p, xgb_p, cat_p):
        self.preprocessor = preprocessor
        self.estimators = [('lgbm', lgb.LGBMRegressor(**lgbm_p)), ('xgb', xgb.XGBRegressor(**xgb_p)),
                           ('catboost', cb.CatBoostRegressor(**cat_p))]
        self.model = StackingRegressor(estimators=self.estimators, final_estimator=LinearRegression(), cv=3, n_jobs=1)

    def fit(self, X_raw, y):
        X_ready = self.preprocessor.transform(X_raw)
        self.model.fit(X_ready, y)
        return self

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        return self.model.predict(X_ready)


def find_optimal_clusters(X_raw, max_clusters):
    # Упрощенная логика поиска (без отрисовки), только расчет
    if len(X_raw) > 100000:
        X_sample = X_raw.sample(50000, random_state=RANDOM_STATE)
    else:
        X_sample = X_raw
    try:
        X_log = np.log2(X_sample.astype(float).clip(lower=1))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_log)
        model = KMeans(random_state=RANDOM_STATE, n_init='auto')
        visualizer = KElbowVisualizer(model, k=(2, max_clusters), metrics='inertia', timings=False)
        visualizer.fit(X_scaled)
        return visualizer.elbow_value_
    except:
        return 3


# --- ОСНОВНОЙ КОД ---
def main():
    print(f"--- ГЕНЕРАЦИЯ ОТЧЕТА ПО БАЗЕ {OPTUNA_DB_NAME} ---")

    # 1. Подключение к Optuna
    storage_url = f"sqlite:///{OPTUNA_DB_NAME}"
    if not os.path.exists(OPTUNA_DB_NAME):
        print(f"❌ ОШИБКА: Файл {OPTUNA_DB_NAME} не найден в папке!")
        return

    try:
        # Пытаемся загрузить исследование
        study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=storage_url)
    except KeyError:
        # Если имя study не угадали, пробуем найти доступные
        print(f"⚠️ Имя study '{OPTUNA_STUDY_NAME}' не найдено. Ищу доступные...")
        studies = optuna.study.get_all_study_summaries(storage=storage_url)
        if not studies:
            print("❌ База данных пуста или повреждена.")
            return
        correct_name = studies[0].study_name
        print(f"✅ Найдено исследование: '{correct_name}'. Использую его.")
        study = optuna.load_study(study_name=correct_name, storage=storage_url)

    best_params = study.best_params
    best_optuna_rmse = study.best_value
    n_trials = len(study.trials)
    print(f"  Загружены параметры (RMSE={best_optuna_rmse:.4f}, Trials={n_trials})")

    # 2. Загрузка данных
    print("--- Загрузка и подготовка данных ---")
    df = pd.read_csv(FILE_PATH, decimal=',', sep=';', header=None, skiprows=1, usecols=COLUMN_MAPPING.keys()).rename(
        columns=COLUMN_MAPPING)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    X_full = df[FEATURE_COLUMNS]
    y_full = df[TARGET_COLUMN]

    # Делим так же, как в основном скрипте
    X_remaining, X_test, y_remaining, y_test = train_test_split(X_full, y_full, test_size=TEST_SIZE_SHAP,
                                                                random_state=RANDOM_STATE)
    # Для финального обучения используем половину от remaining (train часть), чтобы быть честными к методологии
    # Или, если вы в V26 учили на FULL TRAIN (X_train_full), то берем его.
    X_optuna_full, X_train_full, y_optuna_full, y_train_full = train_test_split(X_remaining, y_remaining, test_size=0.5,
                                                                                random_state=RANDOM_STATE)

    # 3. Кластеризация (восстановление)
    print("--- Восстановление кластеров ---")
    k = find_optimal_clusters(X_remaining, MAX_CLUSTERS_TO_TEST)
    if not k: k = 3

    scaler_global = StandardScaler()
    X_rem_log = np.log2(X_remaining.astype(float).clip(lower=1))
    X_rem_scaled = scaler_global.fit_transform(X_rem_log)
    kmeans_global = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
    kmeans_global.fit(X_rem_scaled)
    preprocessor = GlobalPreprocessor(scaler_global, kmeans_global)

    # 4. Сборка модели
    LGBM_BP = {k.replace('lgbm_', ''): v for k, v in best_params.items() if k.startswith('lgbm_')}
    XGB_BP = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
    CAT_BP = {k.replace('cat_', ''): v for k, v in best_params.items() if k.startswith('cat_')}

    # Дефолтные настройки
    LGBM_BP.update({'random_state': RANDOM_STATE, 'n_jobs': 1, 'verbose': -1})
    XGB_BP.update({'random_state': RANDOM_STATE, 'n_jobs': 1})
    CAT_BP.update({'random_state': RANDOM_STATE, 'verbose': 0})

    print("--- Обучение финальной модели... ---")
    model = ProductionModel(preprocessor, LGBM_BP, XGB_BP, CAT_BP)
    # Обучаем на TRAIN части (как в пайплайне)
    model.fit(X_train_full, y_train_full)

    # 5. Расчет метрик
    print("--- Расчет метрик на Test Set ---")
    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)

    # 6. Запись отчета
    print(f"--- Сохранение отчета в {REPORT_FILENAME} ---")

    with open(REPORT_FILENAME, 'w', encoding='utf-8') as f:
        f.write("====================================================\n")
        f.write(f"===   РЕЗУЛЬТАТЫ ИССЛЕДОВАТЕЛЬСКОГО ПАЙПЛАЙНА (V26/V25)   ===\n")
        f.write("====================================================\n\n")

        f.write("--- Общая информация ---\n")
        f.write(f"  Исходные признаки: {FEATURE_COLUMNS}\n")
        f.write(f"  Финальные признаки модели: {FEATURE_COLUMNS + ['cluster_id']}\n")
        f.write(f"  Имя исследования Optuna: {study.study_name}\n")
        f.write(f"  Оптимальное число кластеров (найдено методом локтя): {k}\n")
        f.write(f"  Количество шагов Optuna (N_TRIALS_OPTUNA): {n_trials}\n\n")

        f.write(f"--- Лучшие гиперпараметры, найденные Optuna (RMSE Val={best_optuna_rmse:.4f}) ---\n")
        # Сортируем для красоты
        for key in sorted(best_params.keys()):
            f.write(f"  {key}: {best_params[key]}\n")

        f.write("\n\n")
        f.write(f"--- Метрики качества на тестовой выборке ({TEST_SIZE_SHAP * 100}%) ---\n")
        f.write(f"  R² (Коэффициент детерминации): {r2:.4f}\n")
        f.write(f"  RMSE (Root Mean Squared Error): {rmse:.4f}\n")
        f.write(f"  MAE (Mean Absolute Error):      {mae:.4f}\n")
        f.write(f"  MAPE (Mean Abs. Perc. Error):   {mape:.2%}\n")

    print(f"✅ ГОТОВО! Отчет сохранен: {REPORT_FILENAME}")
    print(f"Preview:\n  R2: {r2:.4f}\n  RMSE: {rmse:.4f}\n  MAE: {mae:.4f}\n  MAPE: {mape:.2%}")


if __name__ == "__main__":
    main()