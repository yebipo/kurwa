# ===================================================================================
# === СКРИПТ ДЛЯ ПРОДВИНУТОЙ НАСТРОЙКИ (V13.0 - ВИЗУАЛИЗАЦИЯ ПОГРЕШНОСТЕЙ) ===
# ===================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# --- 1. КОНФИГУРАЦИЯ ---
FILE_PATH = 'output_adequate.csv'
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5
RANDOM_STATE = 42
N_CLUSTERS = 8
TEST_SIZE = 0.2

# Лучшие параметры от Optuna, которые мы будем использовать для всех моделей
BEST_PARAMS = {
    'n_estimators': 1470, 'learning_rate': 0.088, 'num_leaves': 57,
    'max_depth': 18, 'min_child_samples': 13, 'lambda_l1': 1.64e-07,
    'lambda_l2': 3.64e-08, 'random_state': RANDOM_STATE, 'n_jobs': -1
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
    data = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    if data is None: return
    X_train, X_test, y_train, y_test = data

    print("\nДобавление признака кластеризации (k=8)...")
    X_train_final, X_test_final = add_cluster_features(X_train, X_test, N_CLUSTERS, RANDOM_STATE)

    # --- ЭТАП 1: Обучение трех моделей для квантильной регрессии ---
    quantiles = [0.025, 0.5, 0.975]
    quantile_models = {}

    for q in quantiles:
        print(f"\n--- Обучение модели для квантиля: {q} ---")

        # Создаем модель с нужными параметрами
        params = BEST_PARAMS.copy()
        params['objective'] = 'quantile'  # Указываем, что это квантильная регрессия
        params['alpha'] = q  # Указываем, какой квантиль предсказывать

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train_final, y_train, categorical_feature=['cluster_id'])
        quantile_models[q] = model

    print("\n✅ Все модели для квантилей обучены.")

    # --- ЭТАП 2: Создание предсказаний ---
    print("\n--- Создание предсказаний для границ погрешности ---")
    predictions = pd.DataFrame()
    predictions['y_real'] = y_test

    for q, model in quantile_models.items():
        predictions[f'y_pred_q{q}'] = model.predict(X_test_final)

    # Для красивого графика отсортируем все по реальным значениям
    predictions.sort_values('y_real', inplace=True)
    print("✅ Предсказания созданы и отсортированы.")

    # --- ЭТАП 3: Финальная визуализация ---
    print("\n--- Построение графика с интервалами погрешности ---")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 12))

    # Рисуем идеальную прямую
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.75, zorder=2, label='Идеальная модель (y=x)')

    # Рисуем предсказания медианы (основная линия)
    ax.plot(predictions['y_real'], predictions['y_pred_q0.5'], 'o',
            markersize=5, alpha=0.6, label='Предсказание медианы (50% квантиль)')

    # Закрашиваем область погрешности (95% доверительный интервал)
    ax.fill_between(
        predictions['y_real'],
        predictions['y_pred_q0.025'],
        predictions['y_pred_q0.975'],
        color='skyblue',
        alpha=0.4,
        label='95% интервал погрешности'
    )

    # Настройка графика
    ax.set_title('График "Реальность vs. Предсказание" с интервалами погрешности', fontsize=18)
    ax.set_xlabel('Реальные значения', fontsize=14)
    ax.set_ylabel('Предсказанные значения', fontsize=14)
    ax.set_aspect('equal')
    ax.set_xlim(predictions['y_real'].min() - 0.05, predictions['y_real'].max() + 0.05)
    ax.set_ylim(predictions['y_real'].min() - 0.05, predictions['y_real'].max() + 0.05)
    ax.legend(fontsize=12)

    plt.show()


if __name__ == "__main__":
    main()