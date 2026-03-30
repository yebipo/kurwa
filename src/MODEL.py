# =========================================================================
# ===    ФИНАЛЬНЫЙ СКРИПТ: СОЗДАНИЕ И СОХРАНЕНИЕ ЛУЧШЕЙ МОДЕЛИ         ===
# =========================================================================

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- 1. КОНФИГУРАЦИЯ (Все параметры наших чемпионов) ---
FILE_PATH = 'output_adequate.csv'
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5
RANDOM_STATE = 42
N_CLUSTERS = 8

# Параметры из наших исследований
LGBM_PARAMS = {
    'n_estimators': 1470, 'learning_rate': 0.088, 'num_leaves': 57,
    'max_depth': 18, 'min_child_samples': 13, 'lambda_l1': 1.64e-07,
    'lambda_l2': 3.64e-08, 'random_state': RANDOM_STATE, 'n_jobs': -1
}
XGB_PARAMS = {
    'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 7,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': RANDOM_STATE,
    'n_jobs': -1, 'enable_categorical': True
}
CATBOOST_PARAMS = {
    'iterations': 1000, 'learning_rate': 0.05, 'depth': 7,
    'l2_leaf_reg': 3, 'random_state': RANDOM_STATE, 'verbose': 0
}


# -------------------------------------------------------------------------

class FinalModel:
    """
    Класс-обертка для нашей лучшей модели.
    Включает в себя всю логику предобработки и предсказания.
    """

    def __init__(self, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')

        # Определяем базовые модели для стекинга
        estimators = [
            ('lgbm', lgb.LGBMRegressor(**LGBM_PARAMS)),
            ('xgb', xgb.XGBRegressor(**XGB_PARAMS)),
            ('catboost', cb.CatBoostRegressor(**CATBOOST_PARAMS))
        ]

        # Создаем финальную модель - стекинг
        self.stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=LinearRegression(),
            cv=3,
            n_jobs=1,
            verbose=0
        )
        print("✅ Экземпляр FinalModel создан.")

    def _add_cluster_features(self, X, is_fit=False):
        """Применяет масштабирование и добавляет признак кластера."""
        if is_fit:
            # Обучаем и применяем на тренировочных данных
            X_scaled = self.scaler.fit_transform(X)
            cluster_labels = self.kmeans.fit_predict(X_scaled)
        else:
            # Просто применяем на новых данных
            X_scaled = self.scaler.transform(X)
            cluster_labels = self.kmeans.predict(X_scaled)

        X_with_cluster = X.copy()
        X_with_cluster['cluster_id'] = cluster_labels
        return X_with_cluster

    def fit(self, X, y):
        """Обучает всю модель, включая предобработку."""
        print("--- Начало обучения FinalModel ---")

        # 1. Добавляем признак кластера, обучая scaler и kmeans
        print("   Шаг 1/2: Обучение предобработки (Scaler + KMeans)...")
        X_final = self._add_cluster_features(X, is_fit=True)

        # 2. Обучаем основную модель (стекинг)
        print("   Шаг 2/2: Обучение основной модели (стекинг)...")
        self.stacking_model.fit(X_final, y)

        print("✅ Обучение FinalModel завершено.")
        return self

    def predict(self, X):
        """Делает предсказания на новых данных."""
        # 1. Применяем предобработку к новым данным
        X_final = self._add_cluster_features(X, is_fit=False)

        # 2. Делаем предсказание
        return self.stacking_model.predict(X_final)


def main():
    """Главная функция для обучения и сохранения модели."""

    # 1. Загружаем ВЕСЬ набор данных
    print("--- Загрузка полного набора данных ---")
    try:
        df = pd.read_csv(
            FILE_PATH, decimal=',', sep=';', header=None,
            usecols=FEATURE_COLUMNS + [TARGET_COLUMN], skiprows=1
        )
        df.dropna(subset=[TARGET_COLUMN], inplace=True)
        X_full = df[FEATURE_COLUMNS]
        y_full = df[TARGET_COLUMN]
        print(f"✅ Загружено {len(df)} строк.")
    except Exception as e:
        print(f"❌ ОШИБКА: {e}");
        return

    # 2. Создаем и обучаем нашу финальную модель
    final_model = FinalModel()
    final_model.fit(X_full, y_full)

    # 3. Сохраняем обученную модель в файл
    model_filename = 'the_best_model_ever.joblib'
    print(f"\n--- Сохранение модели в файл '{model_filename}' ---")
    joblib.dump(final_model, model_filename)
    print(f"✅ Модель успешно сохранена!")

    # 4. Проверка: загружаем и делаем тестовое предсказание
    print("\n--- Проверка работоспособности сохраненной модели ---")
    loaded_model = joblib.load(model_filename)
    print("✅ Модель загружена.")

    # Возьмем первые 5 строк из наших данных как пример "новых" данных
    sample_data = X_full.head(5)
    print("\nПример входных данных (первые 5 строк):")
    print(sample_data)

    predictions = loaded_model.predict(sample_data)
    print("\nПредсказания для этих данных:")
    print(predictions)


if __name__ == "__main__":
    main()