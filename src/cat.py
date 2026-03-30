import pandas as pd
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

# ==========================================
# 1. ПОДГОТОВКА ДАННЫХ
# ==========================================
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5


def prepare_data_cat(filepath, is_test=False):
    print(f"-> Загрузка {filepath}...")
    header = None if is_test else 'infer'
    df = pd.read_csv(filepath, sep=';', decimal=',', on_bad_lines='skip', header=header)

    # Принудительно называем колонки f1..f4
    new_names = [f'f{i + 1}' for i in range(len(FEATURE_COLUMNS))] + ['target']
    df = df.iloc[:, FEATURE_COLUMNS + [TARGET_COLUMN]]
    df.columns = new_names

    # Экономная конвертация типов
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.').str.strip(), errors='coerce')
        df[col] = df[col].astype(np.float32)

    y = df['target'].copy()
    X = df.drop(columns=['target'])

    # Базовые временные фичи (проверено: работают лучше всего)
    X['prev_1'] = y.shift(1)
    X['roll_mean_10'] = y.rolling(window=10).mean().shift(1)
    X['roll_std_10'] = y.rolling(window=10).std().shift(1)

    valid_idx = X['roll_std_10'].notna()
    return X[valid_idx].copy(), y[valid_idx].copy()


# ==========================================
# 2. ОБУЧЕНИЕ (MULTI-QUANTILE)
# ==========================================
if __name__ == "__main__":
    try:
        X_train, y_train = prepare_data_cat('million.csv')
        X_test, y_test = prepare_data_cat('unique_rows.csv', is_test=True)

        print(f"\nЗапуск CatBoost MultiQuantile (CPU)...")
        start_time = time.time()

        # Обучаем одну модель на три квантиля сразу!
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='MultiQuantile:alpha=0.05,0.5,0.95',
            random_seed=42,
            verbose=100,
            thread_count=-1
        )

        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

        print(f"-> Обучение завершено за {(time.time() - start_time) / 60:.2f} мин.")

        # ==========================================
        # 3. ПРЕДСКАЗАНИЕ И МЕТРИКИ
        # ==========================================
        # predict вернет матрицу [N, 3], где столбцы - это 5%, 50% и 95%
        preds = model.predict(X_test)
        p_lower = preds[:, 0]
        p_median = preds[:, 1]
        p_upper = preds[:, 2]

        # Расчет покрытия
        coverage = np.mean((y_test >= p_lower) & (y_test <= p_upper)) * 100
        width = np.mean(p_upper - p_lower)

        print(f"\n{'=' * 40}")
        print(f"   ИТОГИ CATBOOST MULTI-QUANTILE")
        print(f"   Фактическое покрытие: {coverage:.2f}%")
        print(f"   Средняя ширина:       {width:.4f}")
        print(f"{'=' * 40}")

        # График
        plt.figure(figsize=(12, 6))
        idx = np.random.choice(len(y_test), 150, replace=False)

        # Сортируем для красоты визуализации
        sort_idx = np.argsort(p_median[idx])
        x = np.arange(150)

        plt.fill_between(x, p_lower[idx][sort_idx], p_upper[idx][sort_idx],
                         color='purple', alpha=0.3, label='Multi-Quantile Corridor (90%)')
        plt.plot(x, p_median[idx][sort_idx], color='purple', linewidth=1, label='Median')
        plt.scatter(x, y_test.iloc[idx].values[sort_idx], color='red', s=15, label='Actual Data')

        plt.title(f"CatBoost MultiQuantile на Unique Rows (Cov: {coverage:.2f}%)")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Критическая ошибка: {e}")
        import traceback

        traceback.print_exc()