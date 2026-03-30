import pandas as pd
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor

# ==========================================
# 1. ПОДГОТОВКА ДАННЫХ (+ DELTA PHYSICS)
# ==========================================
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5


def prepare_data_delta(filepath, is_test=False):
    print(f"-> Загрузка и расчет физики дельт для {filepath}...")
    header = None if is_test else 'infer'
    df = pd.read_csv(filepath, sep=';', decimal=',', on_bad_lines='skip', header=header)

    # Принудительное именование
    new_names = [f'f{i + 1}' for i in range(len(FEATURE_COLUMNS))] + ['target']
    df = df.iloc[:, FEATURE_COLUMNS + [TARGET_COLUMN]]
    df.columns = new_names

    # Конвертация в float32
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.').str.strip(), errors='coerce')
        df[col] = df[col].astype(np.float32)

    y = df['target'].copy()
    X = df.drop(columns=['target'])

    # --- СТАРЫЕ ДОБРЫЕ ФИЧИ ---
    X['prev_1'] = y.shift(1)
    X['roll_mean_10'] = y.rolling(window=10).mean().shift(1)
    X['roll_std_10'] = y.rolling(window=10).std().shift(1)

    # --- НОВИНКА: DELTA PHYSICS (Скорость и Ускорение) ---
    # 1. Скорость изменения энтропии (Velocity)
    X['target_diff_1'] = y.shift(1).diff()
    # 2. Ускорение изменения энтропии (Acceleration)
    X['target_diff_2'] = X['target_diff_1'].diff()

    # 3. Дельты основных настроек (если они меняются, это критично)
    for col in [f'f{i + 1}' for i in range(len(FEATURE_COLUMNS))]:
        X[f'{col}_diff'] = X[col].diff()

    # Чистим NaN после всех дифференциалов
    valid_idx = X['target_diff_2'].notna() & X['roll_std_10'].notna()

    return X[valid_idx].copy(), y[valid_idx].copy()


# ==========================================
# 2. ОБУЧЕНИЕ CATBOOST MULTI-QUANTILE
# ==========================================
if __name__ == "__main__":
    try:
        X_train, y_train = prepare_data_delta('million.csv')
        X_test, y_test = prepare_data_delta('unique_rows.csv', is_test=True)

        print(f"\nДанные готовы. Теперь у нас {X_train.shape[1]} фичей.")
        print("Запуск CatBoost MultiQuantile...")

        start_time = time.time()

        # Обучаем нашего чемпиона
        model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            loss_function='MultiQuantile:alpha=0.05,0.5,0.95',
            random_seed=42,
            verbose=100,
            thread_count=-1
        )

        # Используем unique_rows как валидацию, чтобы не переобучиться
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

        print(f"-> Обучение завершено за {(time.time() - start_time) / 60:.2f} мин.")

        # ==========================================
        # 3. АНАЛИЗ РЕЗУЛЬТАТОВ
        # ==========================================
        preds = model.predict(X_test)
        p_lower, p_median, p_upper = preds[:, 0], preds[:, 1], preds[:, 2]

        coverage = np.mean((y_test >= p_lower) & (y_test <= p_upper)) * 100
        width = np.mean(p_upper - p_lower)

        print(f"\n" + "=" * 40)
        print(f"   РЕЗУЛЬТАТЫ С ДЕЛЬТА-ФИЗИКОЙ")
        print(f"   Фактическое покрытие: {coverage:.2f}%")
        print(f"   Средняя ширина:       {width:.4f}")
        print(f"   Лучший прошлый результат: 86.65% (width 0.0655)")
        print("=" * 40)

        # Посмотрим, какие дельты важнее всего
        importances = model.get_feature_importance()
        feat_importances = pd.Series(importances, index=X_train.columns).sort_values(ascending=False)

        print("\nТоп фичей по важности:")
        print(feat_importances.head(10))

        # График
        plt.figure(figsize=(12, 6))
        idx = np.random.choice(len(y_test), 150, replace=False)
        sort_idx = np.argsort(p_median[idx])
        x = np.arange(150)

        plt.fill_between(x, p_lower[idx][sort_idx], p_upper[idx][sort_idx], color='lime', alpha=0.2,
                         label='Delta-Corridor')
        plt.plot(x, p_median[idx][sort_idx], color='green', linewidth=1)
        plt.scatter(x, y_test.iloc[idx].values[sort_idx], color='red', s=15, alpha=0.6)
        plt.title(f"CatBoost + Delta Physics (Coverage: {coverage:.2f}%)")
        plt.show()

    except Exception as e:
        print(f"Ошибка: {e}")