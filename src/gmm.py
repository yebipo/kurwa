# =========================================================================================
# === ГЕНЕРАТОР ОТЧЕТА (ВАРИАНТ С GMM - Gaussian Mixture Models) ===
# =========================================================================================
import pandas as pd
import numpy as np
import optuna
import os
# Импортируем GMM
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import warnings

# --- НАСТРОЙКИ ---
OPTUNA_DB_NAME = "optuna_v26.db"
OPTUNA_STUDY_NAME = "stacking_v26_boosted"

FILE_PATH = 'output_adequate.csv'
REPORT_FILENAME = "FINAL_REPORT_GMM.txt"  # Изменил имя файла отчета, чтобы не затереть старый
RANDOM_STATE = 42
TEST_SIZE_SHAP = 0.20
MAX_CLUSTERS_TO_TEST = 7

COLUMN_MAPPING = {1: 'feature_1', 2: 'feature_2', 3: 'feature_3', 4: 'feature_4', 5: 'target'}
FEATURE_COLUMNS = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
TARGET_COLUMN = 'target'

warnings.filterwarnings('ignore')


# --- КЛАССЫ ---
class GlobalPreprocessor:
    def __init__(self, scaler, clusterer):
        self.scaler = scaler
        self.clusterer = clusterer  # Теперь здесь может быть и KMeans, и GMM

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        # GMM тоже имеет метод predict (возвращает наиболее вероятный кластер)
        cluster_labels = self.clusterer.predict(X_scaled)
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


def find_optimal_clusters_gmm(X_raw, max_clusters):
    """
    Ищет оптимальное число кластеров для GMM используя критерий BIC.
    (Чем ниже BIC, тем лучше модель описывает данные без лишней сложности)
    """
    print("   > Подбор оптимального k для GMM (по критерию BIC)...")
    if len(X_raw) > 100000:
        X_sample = X_raw.sample(50000, random_state=RANDOM_STATE)
    else:
        X_sample = X_raw

    X_log = np.log2(X_sample.astype(float).clip(lower=1))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    bics = []
    k_range = range(2, max_clusters + 1)

    for k in k_range:
        # n_init=1 для скорости поиска, covariance_type='full' позволяет эллипсы любой формы
        gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=1, covariance_type='full')
        gmm.fit(X_scaled)
        bics.append(gmm.bic(X_scaled))

    # Находим k, где BIC минимален
    best_k = k_range[np.argmin(bics)]
    print(f"   > Лучший BIC найден при k={best_k}")
    return best_k


# --- ОСНОВНОЙ КОД ---
def main():
    print(f"--- ГЕНЕРАЦИЯ ОТЧЕТА (ВЕРСИЯ GMM) ПО БАЗЕ {OPTUNA_DB_NAME} ---")

    # 1. Подключение к Optuna
    storage_url = f"sqlite:///{OPTUNA_DB_NAME}"
    if not os.path.exists(OPTUNA_DB_NAME):
        print(f"❌ ОШИБКА: Файл {OPTUNA_DB_NAME} не найден в папке!")
        return

    try:
        study = optuna.load_study(study_name=OPTUNA_STUDY_NAME, storage=storage_url)
    except KeyError:
        print(f"⚠️ Имя study '{OPTUNA_STUDY_NAME}' не найдено. Ищу доступные...")
        studies = optuna.study.get_all_study_summaries(storage=storage_url)
        if not studies:
            print("❌ База данных пуста.")
            return
        correct_name = studies[0].study_name
        print(f"✅ Найдено исследование: '{correct_name}'. Использую его.")
        study = optuna.load_study(study_name=correct_name, storage=storage_url)

    best_params = study.best_params
    best_optuna_rmse = study.best_value
    n_trials = len(study.trials)

    # 2. Загрузка данных
    print("--- Загрузка и подготовка данных ---")
    df = pd.read_csv(FILE_PATH, decimal=',', sep=';', header=None, skiprows=1, usecols=COLUMN_MAPPING.keys()).rename(
        columns=COLUMN_MAPPING)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors='coerce')
    df.dropna(subset=[TARGET_COLUMN], inplace=True)

    X_full = df[FEATURE_COLUMNS]
    y_full = df[TARGET_COLUMN]

    X_remaining, X_test, y_remaining, y_test = train_test_split(X_full, y_full, test_size=TEST_SIZE_SHAP,
                                                                random_state=RANDOM_STATE)
    X_optuna_full, X_train_full, y_optuna_full, y_train_full = train_test_split(X_remaining, y_remaining, test_size=0.5,
                                                                                random_state=RANDOM_STATE)

    # 3. Кластеризация GMM
    print("--- Кластеризация с помощью GMM (Gaussian Mixture) ---")
    # Используем новую функцию поиска для GMM
    k = find_optimal_clusters_gmm(X_remaining, MAX_CLUSTERS_TO_TEST)

    scaler_global = StandardScaler()
    X_rem_log = np.log2(X_remaining.astype(float).clip(lower=1))
    X_rem_scaled = scaler_global.fit_transform(X_rem_log)

    # ЗАМЕНА KMEANS НА GMM
    # n_init=3 для большей стабильности результата
    gmm_global = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=3, covariance_type='full')
    gmm_global.fit(X_rem_scaled)

    # Передаем GMM вместо KMeans в препроцессор
    preprocessor = GlobalPreprocessor(scaler_global, gmm_global)

    # 4. Сборка модели
    LGBM_BP = {k.replace('lgbm_', ''): v for k, v in best_params.items() if k.startswith('lgbm_')}
    XGB_BP = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
    CAT_BP = {k.replace('cat_', ''): v for k, v in best_params.items() if k.startswith('cat_')}

    LGBM_BP.update({'random_state': RANDOM_STATE, 'n_jobs': 1, 'verbose': -1})
    XGB_BP.update({'random_state': RANDOM_STATE, 'n_jobs': 1})
    CAT_BP.update({'random_state': RANDOM_STATE, 'verbose': 0})

    print("--- Обучение финальной модели... ---")
    model = ProductionModel(preprocessor, LGBM_BP, XGB_BP, CAT_BP)
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
        f.write(f"===   РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА С GMM (V26)   ===\n")
        f.write("====================================================\n\n")

        f.write("--- Общая информация ---\n")
        f.write(f"  Метод кластеризации: Gaussian Mixture Models (GMM)\n")
        f.write(f"  Исходные признаки: {FEATURE_COLUMNS}\n")
        f.write(f"  Оптимальное число кластеров (по критерию BIC): {k}\n")
        f.write(f"  Количество шагов Optuna: {n_trials}\n\n")

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