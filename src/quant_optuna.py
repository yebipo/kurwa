# ==========================================
# 0. ИМПОРТ БИБЛИОТЕК
# ==========================================
import pandas as pd
import numpy as np
import time
import joblib
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

    # 🔥 ИДЕАЛЬНЫЕ ПАРАМЕТРЫ ИЗ БАЗЫ OPTUNA 🔥
    lgbm_params = {
        'n_estimators': 1436,  # Из Optuna
        'learning_rate': 0.023115,  # Из Optuna
        'num_leaves': 178,  # Из Optuna
        'reg_alpha': 0.001019,  # Из Optuna
        'reg_lambda': 0.441791,  # Из Optuna
        'min_child_samples': 300,  # Оставляем 300! Это критично для ширины коридора
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

    # === СОХРАНЕНИЕ МОДЕЛЕЙ ===
    print("\n-> 3. Сохранение пайплайна и моделей в файлы...")
    joblib.dump(cluster_extractor, 'cluster_extractor.joblib')
    joblib.dump({
        'lower': model_lower,
        'median': model_median,
        'upper': model_upper
    }, 'lgbm_quantile_models.joblib')
    print("   [✓] Модели успешно сохранены!")

    # === ПРЕДСКАЗАНИЕ ===
    print("\n-> 4. Предсказание на тесте...")
    preds_lower = model_lower.predict(X_test_eng)
    preds_median = model_median.predict(X_test_eng)
    preds_upper = model_upper.predict(X_test_eng)

    inside_corridor = (y_test >= preds_lower) & (y_test <= preds_upper)
    coverage = np.mean(inside_corridor) * 100

    print(f"\n======================================")
    print(f"=== РЕЗУЛЬТАТЫ КВАНТИЛЬНОЙ ОЦЕНКИ  ===")
    print(f"======================================")
    print(f"Целевое покрытие коридора: 90.00%")
    print(f"Фактическое покрытие:      {coverage:.2f}%")
    print(f"Средняя ширина коридора:   {np.mean(preds_upper - preds_lower):.4f}")

    # === ГРАФИКИ МОДЕЛИ И ВАЖНОСТИ ===
    print("\n-> 5. Отрисовка графиков предсказаний...")
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

    # График 1: Коридор шума
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

    axes[0].fill_between(x_axis, lower_sample, upper_sample, color='lightblue', alpha=0.5, label='Коридор 90%')
    axes[0].plot(x_axis, median_sample, color='blue', linestyle='-', linewidth=2, label='Медиана (50%)')
    axes[0].scatter(x_axis, y_sample, color='red', alpha=0.7, s=20, label='Фактическая энтропия', zorder=5)

    axes[0].set_title('Предсказание границ шума (Top Optuna Params)', fontsize=14)
    axes[0].set_xlabel('Образцы настроек осциллятора (отсортированы)')
    axes[0].set_ylabel('Сгенерированная энтропия')
    axes[0].legend(loc='upper left')

    # График 2: Важность признаков
    feature_names = X_train_eng.columns
    importances = model_median.feature_importances_
    fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance',
                                                                                            ascending=False)

    sns.barplot(x='Importance', y='Feature', data=fi_df, ax=axes[1], palette='magma')
    axes[1].set_title('Важность признаков', fontsize=14)
    axes[1].set_xlabel('Условный вес (использования в деревьях)')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.show()

    # === ГРАФИКИ OPTUNA (ИЗ БАЗЫ ДАННЫХ) ===
    print("\n-> 6. Отрисовка Байесовских распределений Optuna...")
    try:
        import optuna
        from optuna.visualization.matplotlib import plot_param_importances, plot_optimization_history, plot_slice

        # Подключаемся к базе
        study = optuna.load_study(study_name="optuna_lgbm_q_v6_gpu", storage="sqlite:///optuna_lgbm_q_v6_gpu.db")

        # График 1: Какие гиперпараметры влияли сильнее всего на лосс
        plot_param_importances(study)
        plt.title('Optuna: Важность гиперпараметров')
        plt.tight_layout()
        plt.show()

        # График 2: История подбора (сходимость)
        plot_optimization_history(study)
        plt.title('Optuna: История оптимизации')
        plt.tight_layout()
        plt.show()

        # График 3: Распределение (Slice plot - где именно Байес нашел лучшие зоны)
        plot_slice(study)
        plt.suptitle('Optuna: Распределение гиперпараметров (Slice)', y=1.02)
        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Библиотека optuna не установлена. Пропускаем графики подбора.")
    except Exception as e:
        print(f"Не удалось построить графики Optuna: {e}")


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

        # Вектор прошлого (3 шага)
        X['prev_1'] = y.shift(1)
        X['prev_2'] = y.shift(2)
        X['prev_3'] = y.shift(3)

        # Скользящие окна (тренд и волатильность)
        X['roll_mean'] = y.rolling(window=10).mean().shift(1)
        X['roll_std'] = y.rolling(window=10).std().shift(1)

        # Очистка мусора
        valid_idx = ~y.isna() & ~X['roll_std'].isna()
        X = X[valid_idx]
        y = y[valid_idx]

        # Финальные имена (9 штук)
        cols = ['f1', 'f2', 'f3', 'f4', 'prev_1', 'prev_2', 'prev_3', 'roll_mean', 'roll_std']
        X.columns = cols

        return X, y


    print("-> Очистка данных, создание лагов и скользящих окон...")
    X_train, y_train = prepare_sequential_data(train_df, FEATURE_COLUMNS, TARGET_COLUMN)
    X_test, y_test = prepare_sequential_data(test_df, FEATURE_COLUMNS, TARGET_COLUMN)

    print(f"Размер Train: {X_train.shape[0]} строк (используем 9 фичей)")
    print(f"Размер Test:  {X_test.shape[0]} строк\n")

    train_quantile_models(X_train, y_train, X_test, y_test)