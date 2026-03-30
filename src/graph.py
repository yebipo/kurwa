# =========================================================================
# ===      ФИНАЛЬНЫЙ СКРИПТ: ВИЗУАЛИЗАЦИЯ КАЧЕСТВА ЛУЧШЕЙ МОДЕЛИ       ===
# =========================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- 1. КОНФИГУРАЦИЯ (Все параметры наших чемпионов) ---
FILE_PATH = 'output_adequate.csv'
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5
RANDOM_STATE = 42
N_CLUSTERS = 8
TEST_SIZE = 0.2

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
    # Класс FinalModel остается без изменений
    def __init__(self, n_clusters=N_CLUSTERS, random_state=RANDOM_STATE):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init='auto')
        estimators = [
            ('lgbm', lgb.LGBMRegressor(**LGBM_PARAMS)),
            ('xgb', xgb.XGBRegressor(**XGB_PARAMS)),
            ('catboost', cb.CatBoostRegressor(**CATBOOST_PARAMS))
        ]
        self.stacking_model = StackingRegressor(
            estimators=estimators, final_estimator=LinearRegression(), cv=3, n_jobs=1, verbose=0
        )

    def _add_cluster_features(self, X, is_fit=False):
        if is_fit:
            X_scaled = self.scaler.fit_transform(X)
            cluster_labels = self.kmeans.fit_predict(X_scaled)
        else:
            X_scaled = self.scaler.transform(X)
            cluster_labels = self.kmeans.predict(X_scaled)
        X_with_cluster = X.copy()
        X_with_cluster['cluster_id'] = cluster_labels
        return X_with_cluster

    def fit(self, X, y):
        X_final = self._add_cluster_features(X, is_fit=True)
        self.stacking_model.fit(X_final, y)
        return self

    def predict(self, X):
        X_final = self._add_cluster_features(X, is_fit=False)
        return self.stacking_model.predict(X_final)


def main():
    """Главная функция для обучения и ВИЗУАЛИЗАЦИИ модели."""

    # 1. Загружаем и РАЗДЕЛЯЕМ данные
    print("--- Загрузка и разделение данных ---")
    try:
        df = pd.read_csv(
            FILE_PATH, decimal=',', sep=';', header=None,
            usecols=FEATURE_COLUMNS + [TARGET_COLUMN], skiprows=1
        )
        df.dropna(subset=[TARGET_COLUMN], inplace=True)
        X = df[FEATURE_COLUMNS]
        y = df[TARGET_COLUMN]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        print(f"✅ Данные загружены и разделены.")
    except Exception as e:
        print(f"❌ ОШИБКА: {e}");
        return

    # 2. Создаем и обучаем нашу финальную модель на ТРЕНИРОВОЧНЫХ данных
    print("\n--- Обучение финальной модели на тренировочных данных ---")
    final_model = FinalModel()
    final_model.fit(X_train, y_train)

    # 3. Делаем предсказания на ТЕСТОВЫХ данных
    print("\n--- Создание предсказаний для тестовой выборки ---")
    y_pred = final_model.predict(X_test)
    print("✅ Предсказания созданы.")

    # --- НОВЫЙ БЛОК: ВИЗУАЛИЗАЦИЯ ---
    print("\n--- Построение графика 'Реальность vs. Предсказание' ---")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))

    # Рисуем точки
    ax.scatter(y_test, y_pred, alpha=0.5, label='Предсказания модели')

    # Рисуем идеальную прямую y = x
    # Находим минимальное и максимальное значения для красивой линии
    lims = [
        np.min([ax.get_xlim(), ax.get_ylim()]),
        np.max([ax.get_xlim(), ax.get_ylim()]),
    ]
    ax.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Идеальная модель (y=x)')
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Добавляем подписи и заголовок
    ax.set_title('График "Реальность vs. Предсказание" для лучшей модели', fontsize=16)
    ax.set_xlabel('Реальные значения (y_test)', fontsize=12)
    ax.set_ylabel('Предсказанные значения (y_pred)', fontsize=12)
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()