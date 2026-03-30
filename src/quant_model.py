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
    print("-> 1. Генерация признаков (KMeans)...")
    cluster_extractor = ClusterFeatureExtractor(n_clusters=4)
    X_train_eng = cluster_extractor.fit_transform(X_train)
    X_test_eng = cluster_extractor.transform(X_test)

    # Твои гиперпараметры
    lgbm_params = {
        'reg_alpha': 2.6508280629175112e-08,
        'reg_lambda': 0.009910745904773404,
        'learning_rate': 0.029914617989631492,
        'max_depth': 24,
        'min_child_samples': 14,
        'n_estimators': 1037,
        'num_leaves': 192,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': -1
    }

    print("-> 2. Обучение моделей для коридора энтропии...")
    start_time = time.time()

    # Модель для 5% (Нижняя граница)
    model_lower = LGBMRegressor(objective='quantile', alpha=0.05, **lgbm_params)
    model_lower.fit(X_train_eng, y_train)
    print("   [✓] Нижняя граница (5-й перцентиль) обучена")

    # Модель для 50% (Медиана)
    model_median = LGBMRegressor(objective='quantile', alpha=0.50, **lgbm_params)
    model_median.fit(X_train_eng, y_train)
    print("   [✓] Медиана (50-й перцентиль) обучена")

    # Модель для 95% (Верхняя граница)
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
    # Считаем, сколько реальных фактов влезло в наш предсказанный коридор (должно быть около 90%)
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

    # Берем случайные 150 точек для наглядности (иначе график сольется в сплошное пятно)
    sample_size = min(150, len(y_test))
    sample_indices = np.random.choice(len(y_test), sample_size, replace=False)

    y_sample = y_test.iloc[sample_indices].values if isinstance(y_test, pd.Series) else y_test[sample_indices]
    lower_sample = preds_lower[sample_indices]
    median_sample = preds_median[sample_indices]
    upper_sample = preds_upper[sample_indices]

    # Сортируем по медиане для красоты (от низкого шума к высокому)
    sort_idx = np.argsort(median_sample)
    y_sample = y_sample[sort_idx]
    lower_sample = lower_sample[sort_idx]
    median_sample = median_sample[sort_idx]
    upper_sample = upper_sample[sort_idx]
    x_axis = np.arange(sample_size)

    plt.figure(figsize=(14, 6))

    # Рисуем коридор (от 5% до 95%)
    plt.fill_between(x_axis, lower_sample, upper_sample, color='lightblue', alpha=0.5, label='Коридор 90% (5% - 95%)')

    # Рисуем медиану
    plt.plot(x_axis, median_sample, color='blue', linestyle='-', linewidth=2, label='Медиана (50%)')

    # Рисуем реальную энтропию
    plt.scatter(x_axis, y_sample, color='red', alpha=0.7, s=20, label='Фактическая энтропия', zorder=5)

    plt.title('Предсказание границ шума осциллятора (Квантильная регрессия)')
    plt.xlabel('Образцы настроек осциллятора (отсортированы по ожидаемой энтропии)')
    plt.ylabel('Сгенерированная энтропия')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# ==========================================
# 4. ЗАПУСК
# ==========================================
if __name__ == "__main__":
    print("Загрузка данных...")

    train_df = pd.read_csv('million.csv', sep=';', decimal=',', on_bad_lines='skip')
    test_df = pd.read_csv('unique_rows.csv', sep=';', decimal=',', on_bad_lines='skip', header=None)

    # Вырезаем сырые колонки по индексам
    X_train_raw = train_df.iloc[:, FEATURE_COLUMNS]
    y_train_raw = train_df.iloc[:, TARGET_COLUMN]

    X_test_raw = test_df.iloc[:, FEATURE_COLUMNS]
    y_test_raw = test_df.iloc[:, TARGET_COLUMN]


    # 🚑 ФУНКЦИЯ ДЛЯ ПРИНУДИТЕЛЬНОГО ПЕРЕВОДА ТЕКСТА В ЧИСЛА (FLOAT)
    def force_float(data):
        if isinstance(data, pd.Series):
            return pd.to_numeric(data.astype(str).str.replace(',', '.').str.strip(), errors='coerce')
        else:
            return data.apply(
                lambda col: pd.to_numeric(col.astype(str).str.replace(',', '.').str.strip(), errors='coerce'))


    print("-> Жёсткая очистка: конвертация всего в числа...")
    X_train = force_float(X_train_raw)
    y_train = force_float(y_train_raw)
    X_test = force_float(X_test_raw)
    y_test = force_float(y_test_raw)

    # Вычищаем строки, где таргет не смог стать числом и превратился в NaN
    valid_train = ~y_train.isna()
    X_train = X_train[valid_train]
    y_train = y_train[valid_train]

    valid_test = ~y_test.isna()
    X_test = X_test[valid_test]
    y_test = y_test[valid_test]

    # Унифицируем имена колонок
    cols = ['f1', 'f2', 'f3', 'f4']
    X_train.columns = cols
    X_test.columns = cols

    print(f"Размер Train после очистки: {X_train.shape[0]} строк")
    print(f"Размер Test после очистки:  {X_test.shape[0]} строк\n")

    # Погнали!
    train_quantile_models(X_train, y_train, X_test, y_test)