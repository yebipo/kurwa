# =========================================================================================
# ===    ФИНАЛЬНЫЙ СКРИПТ V6: ПОИСК K, OPTUNA, ОБУЧЕНИЕ И ЭКСПОРТ ПРИЗНАКОВ           ===
# ===    (Полный пайплайн: лог-признаки -> локоть -> Optuna -> модель -> экспорт)       ===
# =========================================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
import warnings
import optuna
import matplotlib.pyplot as plt

# Глобальные настройки
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Убираем лишние логи Optuna

# --- 1. КОНФИГУРАЦИЯ ---

FILE_PATH = 'output_adequate.csv'
RANDOM_STATE = 42
N_TRIALS_OPTUNA = 70  # Количество шагов для Optuna (можно уменьшить до 30-40 для ускорения)
FEATURES_EXPORT_PATH = 'data_for_shap_analysis.csv'  # Имя файла для экспорта признаков

# Определяем колонки
COLUMN_MAPPING = {1: 'feature_1', 3: 'feature_3', 5: 'target'}
FEATURE_COLUMNS = ['feature_1', 'feature_3']
TARGET_COLUMN = 'target'
N_CLUSTERS = 8  # Значение по умолчанию, будет обновлено после метода локтя


# --- 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ---

def find_optimal_clusters(X_processed):
    """
    Применяет метод локтя для поиска оптимального k и запрашивает ввод у пользователя.

    Args:
        X_processed (np.array): Данные ПОСЛЕ логарифмирования и масштабирования.
    """
    print("\n--- Поиск оптимального количества кластеров (метод локтя) ---")
    inertia = []
    k_range = range(2, 16)  # Проверяем от 2 до 15 кластеров

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        kmeans.fit(X_processed)
        inertia.append(kmeans.inertia_)
        print(f"  k={k}, inertia={kmeans.inertia_:.2f}")

    # Визуализация
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Метод локтя для определения оптимального k')
    plt.xlabel('Количество кластеров (k)')
    plt.ylabel('Inertia (сумма квадратов расстояний)')
    plt.xticks(k_range)
    plt.grid(True)
    plt.show(block=True)

    print("\nПосмотрите на график и найдите 'локоть' (точку, где график начинает выпрямляться).")
    try:
        optimal_k = int(input("Введите оптимальное количество кластеров: "))
    except (ValueError, TypeError):
        print("Неверный ввод, используется значение по умолчанию 8.")
        optimal_k = 8

    return optimal_k


# --- 3. КЛАСС ФИНАЛЬНОЙ МОДЕЛИ ---

class FinalModel:
    """
    Класс-обертка для модели. Включает всю логику предобработки и предсказания.
    Конфигурируется найденными оптимальными параметрами.
    """

    def __init__(self, lgbm_params, xgb_params, catboost_params, n_clusters, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        self.feature_names_in_ = None

        estimators = [
            ('lgbm', lgb.LGBMRegressor(**lgbm_params)),
            ('xgb', xgb.XGBRegressor(**xgb_params)),
            ('catboost', cb.CatBoostRegressor(**catboost_params))
        ]
        self.stacking_model = StackingRegressor(
            estimators=estimators, final_estimator=LinearRegression(),
            cv=3, n_jobs=-1, verbose=0
        )

    def _log_transform(self, X):
        """Преобразует признаки в логарифмическую шкалу (log2)."""
        X_transformed = X.copy()
        for col in X_transformed.columns:
            X_transformed[col] = np.log2(X_transformed[col].astype(float).clip(lower=1))
        return X_transformed

    def _add_cluster_features(self, X_log_transformed, is_fit=False):
        """Применяет масштабирование и добавляет признак кластера к лог. данным."""
        if is_fit:
            X_scaled = self.scaler.fit_transform(X_log_transformed)
            cluster_labels = self.kmeans.fit_predict(X_scaled)
        else:
            X_scaled = self.scaler.transform(X_log_transformed)
            cluster_labels = self.kmeans.predict(X_scaled)

        X_with_cluster = X_log_transformed.copy()
        X_with_cluster['cluster_id'] = cluster_labels
        return X_with_cluster

    def fit(self, X_raw, y):
        """Обучает весь пайплайн на сырых данных."""
        self.feature_names_in_ = X_raw.columns.tolist()
        X_log = self._log_transform(X_raw)
        X_final = self._add_cluster_features(X_log, is_fit=True)
        self.stacking_model.fit(X_final, y)
        return self

    def predict(self, X_raw):
        """Делает предсказания на сырых данных."""
        if X_raw.columns.tolist() != self.feature_names_in_:
            raise ValueError(
                f"Неверные имена колонок! Модель обучалась на {self.feature_names_in_}, а получила {X_raw.columns.tolist()}")
        X_log = self._log_transform(X_raw)
        X_final = self._add_cluster_features(X_log, is_fit=False)
        return self.stacking_model.predict(X_final)

    def get_final_features(self, X_raw):
        """Применяет пайплайн предобработки и возвращает финальные признаки для анализа."""
        # Убедимся, что scaler и kmeans обучены
        if not hasattr(self.scaler, 'mean_'):
            raise RuntimeError("Модель еще не обучена. Вызовите .fit() перед get_final_features().")
        X_log = self._log_transform(X_raw)
        X_final_features = self._add_cluster_features(X_log, is_fit=False)
        return X_final_features


# --- 4. ФУНКЦИЯ ДЛЯ ОПТИМИЗАЦИИ В OPTUNA ---

def objective(trial, X_processed, y_target):
    """Целевая функция, которую Optuna будет минимизировать."""
    lgbm_params = {
        'n_estimators': trial.suggest_int('lgbm_n_estimators', 400, 2000),
        'learning_rate': trial.suggest_float('lgbm_learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('lgbm_num_leaves', 20, 150),
        'max_depth': trial.suggest_int('lgbm_max_depth', 5, 20),
        'min_child_samples': trial.suggest_int('lgbm_min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lgbm_lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lgbm_lambda_l2', 1e-8, 10.0, log=True),
        'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1
    }
    xgb_params = {
        'n_estimators': trial.suggest_int('xgb_n_estimators', 400, 2000),
        'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
        'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
        'random_state': RANDOM_STATE, 'n_jobs': -1
    }
    catboost_params = {
        'iterations': trial.suggest_int('cat_iterations', 400, 2000),
        'learning_rate': trial.suggest_float('cat_learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('cat_depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('cat_l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_state': RANDOM_STATE, 'verbose': 0
    }

    estimators = [
        ('lgbm', lgb.LGBMRegressor(**lgbm_params)),
        ('xgb', xgb.XGBRegressor(**xgb_params)),
        ('catboost', cb.CatBoostRegressor(**catboost_params))
    ]
    stacking_model = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=3, n_jobs=-1)

    score = cross_val_score(stacking_model, X_processed, y_target, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    return -np.mean(score)


# --- 5. ГЛАВНЫЙ СКРИПТ ---

def main():
    # --- ШАГ 1: Загрузка данных ---
    print("--- Шаг 1/6: Загрузка и предварительная обработка данных ---")
    try:
        df = pd.read_csv(FILE_PATH, decimal=',', sep=';', header=None, usecols=COLUMN_MAPPING.keys(),
                         skiprows=1).rename(columns=COLUMN_MAPPING)
        df.dropna(subset=[TARGET_COLUMN], inplace=True)
        X_raw = df[FEATURE_COLUMNS]
        y_full = df[TARGET_COLUMN]
    except Exception as e:
        print(f"❌ ОШИБКА ЗАГРУЗКИ ДАННЫХ: {e}");
        return

    # --- ШАГ 2: Поиск оптимального K для кластеризации ---
    print("\n--- Шаг 2/6: Поиск оптимального количества кластеров ---")
    temp_model_for_elbow = FinalModel({}, {}, {}, n_clusters=2)  # Временный экземпляр для доступа к методам
    X_log_for_elbow = temp_model_for_elbow._log_transform(X_raw)
    scaler_for_elbow = StandardScaler()
    X_processed_for_elbow = scaler_for_elbow.fit_transform(X_log_for_elbow)

    global N_CLUSTERS
    N_CLUSTERS = find_optimal_clusters(X_processed_for_elbow)
    print(f"✅ Выбрано оптимальное количество кластеров: {N_CLUSTERS}")

    # --- ШАГ 3: Запуск поиска параметров с Optuna ---
    print(f"\n--- Шаг 3/6: Запуск Optuna ({N_TRIALS_OPTUNA} шагов) ---")
    print("Это может занять значительное время...")

    # Подготавливаем финальные данные для Optuna с новым N_CLUSTERS
    # Создаем временный обработчик, чтобы получить признаки для Optuna
    temp_preprocessor = FinalModel({}, {}, {}, n_clusters=N_CLUSTERS)
    X_for_optuna = temp_preprocessor._add_cluster_features(temp_preprocessor._log_transform(X_raw), is_fit=True)

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_for_optuna, y_full), n_trials=N_TRIALS_OPTUNA, n_jobs=1)

    print("\n✅ Поиск параметров завершен!")
    print(f"  Лучшее значение MSE на кросс-валидации: {study.best_value:.6f}")

    # --- ШАГ 4: Обучение финальной модели ---
    print("\n--- Шаг 4/6: Обучение финальной модели на лучших параметрах ---")
    best_params = study.best_params
    LGBM_BEST_PARAMS = {k.replace('lgbm_', ''): v for k, v in best_params.items() if k.startswith('lgbm_')}
    XGB_BEST_PARAMS = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
    CATBOOST_BEST_PARAMS = {k.replace('cat_', ''): v for k, v in best_params.items() if k.startswith('cat_')}

    LGBM_BEST_PARAMS.update({'random_state': RANDOM_STATE, 'n_jobs': -1, 'verbose': -1})
    XGB_BEST_PARAMS.update({'random_state': RANDOM_STATE, 'n_jobs': -1})
    CATBOOST_BEST_PARAMS.update({'random_state': RANDOM_STATE, 'verbose': 0})

    final_tuned_model = FinalModel(
        lgbm_params=LGBM_BEST_PARAMS,
        xgb_params=XGB_BEST_PARAMS,
        catboost_params=CATBOOST_BEST_PARAMS,
        n_clusters=N_CLUSTERS
    )

    final_tuned_model.fit(X_raw, y_full)

    # --- ШАГ 5: Сохранение модели ---
    model_filename = 'the_best_model_v6_final.joblib'
    print(f"\n--- Шаг 5/6: Сохранение финальной модели в файл '{model_filename}' ---")
    joblib.dump(final_tuned_model, model_filename)
    print(f"✅ Модель успешно сохранена!")

    # --- ШАГ 6: Экспорт обработанных признаков для анализа ---
    print(f"\n--- Шаг 6/6: Экспорт финальных признаков в '{FEATURES_EXPORT_PATH}' ---")

    final_features_df = final_tuned_model.get_final_features(X_raw)
    final_features_df.to_csv(FEATURES_EXPORT_PATH, index=False)
    print(f"✅ Признаки ({final_features_df.shape[1]} колонок) успешно сохранены.")
    print("   Этот файл можно использовать для SHAP-анализа новой модели.")

    print("\n--- Скрипт полностью завершен ---")


if __name__ == "__main__":
    main()