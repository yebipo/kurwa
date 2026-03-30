# =========================================================================
# ===    ФИНАЛЬНЫЙ СКРИПТ: ПАНЕЛЬ ВИЗУАЛИЗАЦИИ КАЧЕСТВА МОДЕЛИ         ===
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

def load_and_prepare_data(file_path, feature_cols, target_col, test_size, random_state):
    # Эта функция без изменений
    print(f"\n--- Загрузка и подготовка данных ---")
    try:
        df = pd.read_csv(file_path, decimal=',', sep=';', header=None, usecols=feature_cols + [target_col], skiprows=1)
    except Exception as e:
        print(f"❌ ОШИБКА: {e}");
        return None
    df.dropna(subset=[target_col], inplace=True)
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"✅ Данные разделены.")
    return X_train, X_test, y_train, y_test


def add_cluster_features(X_train, X_test, n_clusters, random_state):
    # Эта функция без изменений
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    X_train_with_cluster = X_train.copy()
    X_train_with_cluster['cluster_id'] = kmeans.fit_predict(X_train_scaled)
    X_test_with_cluster = X_test.copy()
    X_test_with_cluster['cluster_id'] = kmeans.predict(X_test_scaled)
    return X_train_with_cluster, X_test_with_cluster


def main():
    """Главная функция для обучения и ВИЗУАЛИЗАЦИИ модели."""

    data = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    if data is None: return
    X_train, X_test, y_train, y_test = data

    print("\nДобавление признака кластеризации (k=8)...")
    X_train_final, X_test_final = add_cluster_features(X_train, X_test, N_CLUSTERS, RANDOM_STATE)

    print("\n--- Обучение лучшей модели (Стекинг Титанов) ---")
    estimators = [
        ('lgbm', lgb.LGBMRegressor(**LGBM_PARAMS)),
        ('xgb', xgb.XGBRegressor(**XGB_PARAMS)),
        ('catboost', cb.CatBoostRegressor(**CATBOOST_PARAMS))
    ]
    stacking_regressor = StackingRegressor(
        estimators=estimators, final_estimator=LinearRegression(), cv=3, n_jobs=1, verbose=1
    )
    stacking_regressor.fit(X_train_final, y_train)

    print("\n--- Создание предсказаний для тестовой выборки ---")
    y_pred = stacking_regressor.predict(X_test_final)
    print("✅ Предсказания созданы.")

    # --- НОВЫЙ БЛОК: ВИЗУАЛИЗАЦИЯ ПАНЕЛИ ИЗ ДВУХ ГРАФИКОВ ---
    print("\n--- Построение панели с графиками анализа погрешностей ---")

    plt.style.use('seaborn-v0_8-whitegrid')
    # Создаем фигуру с двумя под-графиками (1 ряд, 2 колонки)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle('Анализ погрешностей предсказаний модели', fontsize=20)

    # --- График 1: Предсказания vs. Реальные значения ---
    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.4, s=20, edgecolors='w')
    lims = [
        np.min([ax1.get_xlim(), ax1.get_ylim()]),
        np.max([ax1.get_xlim(), ax1.get_ylim()]),
    ]
    ax1.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Идеальное предсказание')
    ax1.set_aspect('equal')
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_title('Предсказания vs. Реальные значения', fontsize=16)
    ax1.set_xlabel('Реальные значения', fontsize=12)
    ax1.set_ylabel('Предсказанные значения', fontsize=12)
    ax1.legend()

    # --- График 2: График погрешностей (Residual Plot) ---
    ax2 = axes[1]
    residuals = y_test - y_pred  # Ошибка = Реальное - Предсказанное
    ax2.scatter(y_pred, residuals, alpha=0.4, s=20, edgecolors='w')
    ax2.axhline(y=0, color='r', linestyle='--', label='Нулевая погрешность')
    ax2.set_title('График погрешностей (Residual Plot)', fontsize=16)
    ax2.set_xlabel('Предсказанные значения', fontsize=12)
    ax2.set_ylabel('Погрешность (Реальное - Предсказанное)', fontsize=12)
    ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Немного пододвинем графики, чтобы заголовок не наезжал
    plt.show()


if __name__ == "__main__":
    main()