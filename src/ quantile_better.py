# ==========================================
# 0. ИМПОРТ БИБЛИОТЕК
# ==========================================
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from lightgbm import LGBMRegressor

sns.set_theme(style="whitegrid")

# ==========================================
# 1. НАСТРОЙКИ ДАННЫХ
# ==========================================
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5


# ==========================================
# 2. ФУНКЦИЯ КЛАСТЕРИЗАЦИИ
# ==========================================
class ClusterFeatureExtractor:
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=random_state)

    def fit(self, X):
        self.kmeans.fit(X)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy['cluster_id'] = self.kmeans.predict(X)
        return X_copy

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)


# ==========================================
# 3. ОСНОВНОЙ ПАЙПЛАЙН (КВАНТИЛИ)
# ==========================================
def train_quantile_models(X_train, y_train, X_test, y_test):
    print("-> 1. Генерация признака cluster_id (KMeans)...")
    cluster_extractor = ClusterFeatureExtractor(n_clusters=4)
    X_train_eng = cluster_extractor.fit_transform(X_train)
    X_test_eng = cluster_extractor.transform(X_test)

    # ПАРАМЕТРЫ ДЛЯ ОЦЕНКИ РАЗБРОСА
    lgbm_params = {
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_samples': 300,
        'n_estimators': 150,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    print("-> 2. Обучение моделей для коридора энтропии...")
    start_time = time.time()

    model_lower = LGBMRegressor(objective='quantile', alpha=0.05, **lgbm_params)
    model_lower.fit(X_train_eng, y_train)
    print("   [✓] Нижняя граница (5-й перцентиль) обучена")

    model_median = LGBMRegressor(objective='quantile', alpha=0.50, **lgbm_params)
    model_median.fit(X_train_eng, y_train)
    print("   [✓] Медиана (50-й перцентиль) обучена")

    model_upper = LGBMRegressor(objective='quantile', alpha=0.95, **lgbm_params)
    model_upper.fit(X_train_eng, y_train)
    print("   [✓] Верхняя граница (95-й перцентиль) обучена")

    print(f"-> Обучение завершено за {(time.time() - start_time) / 60:.2f} минут.")

    # === ПРЕДСКАЗАНИЕ ===
    print("-> 3. Предсказание на тесте...")
    preds_lower = model_lower.predict(X_test_eng)
    preds_median = model_median.predict(X_test_eng)
    preds_upper = model_upper.predict(X_test_eng)

    # === МЕТРИКА: COVERAGE (ПОКРЫТИЕ) ===
    inside_corridor = (y_test >= preds_lower) & (y_test <= preds_upper)
    coverage = np.mean(inside_corridor) * 100

    print(f"\n======================================")
    print(f"=== РЕЗУЛЬТАТЫ КВАНТИЛЬНОЙ ОЦЕНКИ  ===")
    print(f"======================================")
    print(f"Целевое покрытие коридора: 90.00%")
    print(f"Фактическое покрытие:      {coverage:.2f}%")
    print(f"Средняя ширина коридора:   {np.mean(preds_upper - preds_lower):.4f}")

    # === ГРАФИКИ ===
    print("\n-> 4. Отрисовка графиков...")

    sample_size = min(150, len(y_test))
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)

    y_sample = y_test.iloc[sample_indices].values if isinstance(y_test, pd.Series) else y_test[sample_indices]
    lower_sample = preds_lower[sample_indices]
    median_sample = preds_median[sample_indices]
    upper_sample = preds_upper[sample_indices]

    sort_idx = np.argsort(median_sample)
    y_sample = y_sample[sort_idx]
    lower_sample = lower_sample[sort_idx]
    median_sample = median_sample[sort_idx]
    upper_sample = upper_sample[sort_idx]
    x_axis = np.arange(sample_size)

    plt.figure(figsize=(14, 6))

    plt.fill_between(x_axis, lower_sample, upper_sample, color='lightblue', alpha=0.5, label='Коридор 90% (5% - 95%)')
    plt.plot(x_axis, median_sample, color='blue', linestyle='-', linewidth=2, label='Медиана (50%)')
    plt.scatter(x_axis, y_sample, color='red', alpha=0.7, s=20, label='Фактическая энтропия', zorder=5)

    plt.title('Предсказание границ шума (Лаги + Скользящие окна)')
    plt.xlabel('Образцы настроек осциллятора (отсортированы по ожидаемой энтропии)')
    plt.ylabel('Сгенерированная энтропия')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. ЗАПУСК И МАГИЯ ВРЕМЕННЫХ РЯДОВ
# ==========================================
if __name__ == "__main__":
    print("Загрузка данных...")

    train_df = pd.read_csv('million.csv', sep=';', decimal=',', on_bad_lines='skip')
    test_df = pd.read_csv('unique_rows.csv', sep=';', decimal=',', on_bad_lines='skip', header=None)


    def prepare_sequential_data(df, feature_cols, target_col):
        X_raw = df.iloc[:, feature_cols].copy()
        y_raw = df.iloc[:, target_col].copy()

        def force_float(data):
            if isinstance(data, pd.Series):
                return pd.to_numeric(data.astype(str).str.replace(',', '.').str.strip(), errors='coerce')
            else:
                return data.apply(
                    lambda col: pd.to_numeric(col.astype(str).str.replace(',', '.').str.strip(), errors='coerce'))

        X = force_float(X_raw)
        y = force_float(y_raw)

        # 🚀 1. Вектор недавнего прошлого (3 шага)
        X['prev_1'] = y.shift(1)
        X['prev_2'] = y.shift(2)
        X['prev_3'] = y.shift(3)

        # 🔥 2. ФИЗИКА ПРОЦЕССА: Скользящие окна за 10 шагов (обязательно shift(1)!)
        # Тренд (нагрев) и волатильность (джиттер)
        X['roll_mean'] = y.rolling(window=10).mean().shift(1)
        X['roll_std'] = y.rolling(window=10).std().shift(1)

        # Удаляем первые 10 строк, так как для них еще не накопилось окно
        valid_idx = ~y.isna() & ~X['roll_std'].isna()
        X = X[valid_idx]
        y = y[valid_idx]

        # Переименовываем колонки (теперь у нас 9 фичей!)
        cols = ['f1', 'f2', 'f3', 'f4', 'prev_1', 'prev_2', 'prev_3', 'roll_mean', 'roll_std']
        X.columns = cols

        return X, y


    print("-> Очистка данных, создание лагов и скользящих окон...")
    X_train, y_train = prepare_sequential_data(train_df, FEATURE_COLUMNS, TARGET_COLUMN)
    X_test, y_test = prepare_sequential_data(test_df, FEATURE_COLUMNS, TARGET_COLUMN)

    print(f"Размер Train: {X_train.shape[0]} строк (используем 9 фичей)")
    print(f"Размер Test:  {X_test.shape[0]} строк\n")

    train_quantile_models(X_train, y_train, X_test, y_test)