# ==========================================
# 0. ИМПОРТ БИБЛИОТЕК
# ==========================================
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor, early_stopping, log_evaluation

sns.set_theme(style="whitegrid")

# ==========================================
# 1. НАСТРОЙКИ ДАННЫХ
# ==========================================
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5


# ==========================================
# 2. ОСНОВНОЙ ПАЙПЛАЙН (СПАРТАНСКИЙ ЧЕМПИОН)
# ==========================================
def train_quantile_models(X_train, y_train, X_test, y_test):
    # 🛠️ Хронологический сплит для ранней остановки (последние 10%)
    split_idx = int(len(X_train) * 0.9)
    X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    # 🔥 ИДЕАЛЬНЫЕ ПАРАМЕТРЫ "СПАРТАНСКОЙ" OPTUNA V8 🔥
    lgbm_params = {
        'learning_rate': 0.056371,
        'max_depth': 5,  # [!] Деревья-коротышки
        'num_leaves': 50,
        'min_child_samples': 600,  # [!] Защита от мелкого шума
        'n_estimators': 250,  # [!] Жесткий лимит на количество деревьев
        'reg_alpha': 0.522038,
        'reg_lambda': 0.888890,
        'subsample': 0.440219,  # [!] Мощный Bagging
        'subsample_freq': 1,
        'random_state': 42,
        'n_jobs': -1
    }

    print("-> 1. Обучение моделей Спартанца с Early Stopping...")
    start_time = time.time()

    callbacks = [early_stopping(stopping_rounds=50), log_evaluation(0)]

    # --- НИЖНЯЯ ГРАНИЦА (Честные 5%) ---
    model_lower = LGBMRegressor(objective='quantile', alpha=0.05, **lgbm_params)
    model_lower.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)

    # --- МЕДИАНА (50%) ---
    model_median = LGBMRegressor(objective='quantile', alpha=0.50, **lgbm_params)
    model_median.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)

    # --- ВЕРХНЯЯ ГРАНИЦА (Честные 95%) ---
    model_upper = LGBMRegressor(objective='quantile', alpha=0.95, **lgbm_params)
    model_upper.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=callbacks)

    print(f"-> Обучение завершено за {(time.time() - start_time) / 60:.2f} минут.")

    # === ПРЕДСКАЗАНИЕ ===
    print("\n-> 2. Предсказание на тесте...")
    preds_lower = model_lower.predict(X_test)
    preds_median = model_median.predict(X_test)
    preds_upper = model_upper.predict(X_test)

    # === МЕТРИКА: COVERAGE ===
    inside_corridor = (y_test >= preds_lower) & (y_test <= preds_upper)
    coverage = np.mean(inside_corridor) * 100
    width = np.mean(preds_upper - preds_lower)

    print(f"\n======================================")
    print(f"=== РЕЗУЛЬТАТЫ (СПАРТАНСКАЯ ОПТЮНА) ===")
    print(f"======================================")
    print(f"Целевое покрытие коридора: 90.00%")
    print(f"Фактическое покрытие:      {coverage:.2f}%")
    print(f"Средняя ширина коридора:   {width:.4f}")

    # === ГРАФИКИ ===
    print("\n-> 3. Отрисовка графиков...")

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))

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

    axes[0].set_title('Предсказание границ шума (Спартанская Модель)', fontsize=14)
    axes[0].set_xlabel('Образцы настроек осциллятора (отсортированы)')
    axes[0].set_ylabel('Сгенерированная энтропия')
    axes[0].legend(loc='upper left')

    feature_names = X_train.columns
    importances = model_median.feature_importances_

    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    sns.barplot(x='Importance', y='Feature', data=fi_df, ax=axes[1], palette='magma')
    axes[1].set_title('Важность признаков (Что сильнее всего влияет на шум?)', fontsize=14)
    axes[1].set_xlabel('Условный вес')
    axes[1].set_ylabel('')

    plt.tight_layout()
    plt.show()


# ==========================================
# 3. ПОДГОТОВКА ДАННЫХ И ЗАПУСК
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

        X['prev_1'] = y.shift(1)
        X['prev_2'] = y.shift(2)
        X['prev_3'] = y.shift(3)
        X['roll_mean'] = y.rolling(window=10).mean().shift(1)
        X['roll_std'] = y.rolling(window=10).std().shift(1)

        valid_idx = ~y.isna() & ~X['roll_std'].isna()
        X = X[valid_idx]
        y = y[valid_idx]

        cols = ['f1', 'f2', 'f3', 'f4', 'prev_1', 'prev_2', 'prev_3', 'roll_mean', 'roll_std']
        X.columns = cols

        return X, y


    print("-> Очистка данных, создание лагов и скользящих окон...")
    X_train, y_train = prepare_sequential_data(train_df, FEATURE_COLUMNS, TARGET_COLUMN)
    X_test, y_test = prepare_sequential_data(test_df, FEATURE_COLUMNS, TARGET_COLUMN)

    print(f"Размер Train: {X_train.shape[0]} строк")
    print(f"Размер Test:  {X_test.shape[0]} строк\n")

    train_quantile_models(X_train, y_train, X_test, y_test)