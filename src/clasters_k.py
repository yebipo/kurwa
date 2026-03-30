# ===================================================================================
# === СКРИПТ ДЛЯ ПРОДВИНУТОЙ НАСТРОЙКИ (V10.0 - ОПТИМИЗАЦИЯ КЛАСТЕРОВ) ===
# ===================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import time
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# --- 1. КОНФИГУРАЦИЯ ---
FILE_PATH = 'output_adequate.csv'
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5
TEST_SIZE = 0.2
RANDOM_STATE = 42

BEST_LGBM_PARAMS = {
    'n_estimators': 1200, 'learning_rate': 0.1, 'num_leaves': 45,
    'random_state': RANDOM_STATE, 'n_jobs': -1
}


# -------------------------------------------------------------------------

def load_and_prepare_data(file_path, feature_cols, target_col, test_size, random_state):
    """Загружает и подготавливает данные."""
    print(f"\n--- Загрузка данных ---")
    try:
        df = pd.read_csv(
            file_path, decimal=',', sep=';', header=None,
            usecols=feature_cols + [target_col], skiprows=1
        )
    except Exception as e:
        print(f"❌ ОШИБКА: {e}");
        return None

    df.dropna(subset=[target_col], inplace=True)
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"✅ Данные разделены.")
    return X_train, X_test, y_train, y_test


def evaluate_all_metrics(y_true, y_pred, model_name=""):
    """Считает и выводит все ключевые метрики."""
    # Эта функция остается без изменений
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_true_for_mape = y_true.copy()
    y_true_for_mape[y_true_for_mape == 0] = 1e-6
    mape = np.mean(np.abs((y_true - y_pred) / y_true_for_mape)) * 100

    print(f"\n--- Метрики для модели: {model_name} ---")
    print(f"  R-squared: {r2:.6f}, MAE: {mae:.6f}, RMSE: {rmse:.6f}, MAPE: {mape:.2f}%")
    return {'R-squared': r2, 'MAE': mae, 'RMSE': rmse, 'MAPE (%)': mape}


def run_cluster_experiment(X_train, X_test, y_train, y_test, n_clusters):
    """Проводит эксперимент с добавлением признака кластеризации."""
    model_name = f'LGBM + Кластеры (k={n_clusters})'
    print(f"\n" + "=" * 50)
    print(f"--- ЭКСПЕРИМЕНТ: {model_name} ---")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"   Обучение KMeans на {n_clusters} кластеров...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')

    train_cluster_labels = kmeans.fit_predict(X_train_scaled)
    X_train_with_cluster = X_train.copy()
    X_train_with_cluster['cluster_id'] = train_cluster_labels

    test_cluster_labels = kmeans.predict(X_test_scaled)
    X_test_with_cluster = X_test.copy()
    X_test_with_cluster['cluster_id'] = test_cluster_labels
    print("   Признак 'cluster_id' добавлен.")

    print("   Обучение LightGBM на данных с кластерами...")
    model = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
    model.fit(X_train_with_cluster, y_train, categorical_feature=['cluster_id'])

    y_pred = model.predict(X_test_with_cluster)
    return evaluate_all_metrics(y_test, y_pred, model_name)


def main():
    all_results = {}

    data = load_and_prepare_data(FILE_PATH, FEATURE_COLUMNS, TARGET_COLUMN, TEST_SIZE, RANDOM_STATE)
    if data is None: return
    X_train, X_test, y_train, y_test = data

    # --- ЭТАП 1: Метод локтя для поиска оптимального k ---
    print("\n" + "=" * 50)
    print("--- ЭТАП 1: Поиск оптимального k с помощью 'метода локтя' ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    inertia = []
    k_range = range(2, 16)
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init='auto')
        kmeans.fit(X_train_scaled)
        inertia.append(kmeans.inertia_)
        print(f"  k={k}, инерция={kmeans.inertia_:.2f}")

    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia, marker='o')
    plt.title('Метод локтя для K-Means', fontsize=16)
    plt.xlabel('Количество кластеров (k)', fontsize=12)
    plt.ylabel('Инерция (Within-cluster sum of squares)', fontsize=12)
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

    # --- ЭТАП 2: Обучение модели-чемпиона (бенчмарк) ---
    print("\n--- ЭТАП 2: Обучение модели-чемпиона (LightGBM Normal) ---")
    champion_model = lgb.LGBMRegressor(**BEST_LGBM_PARAMS)
    champion_model.fit(X_train, y_train)
    y_pred_champion = champion_model.predict(X_test)
    all_results['LightGBM (Чемпион)'] = evaluate_all_metrics(y_test, y_pred_champion, "LightGBM (Чемпион)")

    # --- ЭТАП 3: Финальный тест с разными k ---
    # Судя по прошлому запуску, оптимум где-то около 5. Проверим диапазон.
    k_values_to_test = [3, 4, 5, 6, 7, 8]
    for k in k_values_to_test:
        all_results[f'LGBM + Кластеры (k={k})'] = run_cluster_experiment(X_train.copy(), X_test.copy(), y_train, y_test,
                                                                         n_clusters=k)

    # --- ЭТАП 4: Итоговое сравнение ---
    print("\n" + "=" * 50)
    print("###          ФИНАЛЬНОЕ СРАВНЕНИЕ ВСЕХ ПОДХОДОВ        ###")
    print("=" * 50)

    results_df = pd.DataFrame(all_results).T
    results_df = results_df.sort_values(by='RMSE', ascending=True)

    formatters = {
        'R-squared': '{:,.6f}'.format, 'MAE': '{:,.6f}'.format,
        'RMSE': '{:,.6f}'.format, 'MAPE (%)': '{:,.2f}%'.format
    }
    print(results_df.to_string(formatters=formatters))

    best_model_name = results_df.index[0]
    print(f"\n🏆🏆🏆 Абсолютный чемпион: '{best_model_name}'")


if __name__ == "__main__":
    main()