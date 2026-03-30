# ==========================================
# 0. ИМПОРТ БИБЛИОТЕК
# ==========================================
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

sns.set_theme(style="whitegrid")

# ==========================================
# 1. НАСТРОЙКИ И ПАРАМЕТРЫ
# ==========================================
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5

# Наши лучшие "ручные" настройки
LGBM_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_samples': 300,
    'n_estimators': 150,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

XGB_PARAMS = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 200,
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}


# ==========================================
# 2. ФУНКЦИЯ ПОДГОТОВКИ ДАННЫХ
# ==========================================
def prepare_sequential_data(df, feature_cols, target_col, is_test=False):
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

    X.columns = ['f1', 'f2', 'f3', 'f4', 'prev_1', 'prev_2', 'prev_3', 'roll_mean', 'roll_std']
    return X, y


# ==========================================
# 3. СТЕКИНГ / BLENDING
# ==========================================
def train_stacking_ensemble(X_train, y_train, X_test, y_test):
    # Разделяем обучающую выборку: 80% на базовые модели, 20% на мета-модель
    split_idx = int(len(X_train) * 0.8)
    X_l0, X_meta = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
    y_l0, y_meta = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

    quantiles = [0.05, 0.50, 0.95]
    final_predictions = {}

    print(f"-> Начинаем сборку ансамбля для {len(quantiles)} квантилей...")
    start_time = time.time()

    for q in quantiles:
        print(f"   [~] Обработка квантиля {q}...")

        # 1. Обучаем базовые модели (Level 0)
        m_lgbm = LGBMRegressor(objective='quantile', alpha=q, **LGBM_PARAMS)
        m_lgbm.fit(X_l0, y_l0)

        m_xgb = XGBRegressor(objective='reg:quantileerror', quantile_alpha=q, **XGB_PARAMS)
        m_xgb.fit(X_l0, y_l0)

        # 2. Генерируем мета-признаки (предсказания на части для мета-обучения)
        p_lgbm_meta = m_lgbm.predict(X_meta)
        p_xgb_meta = m_xgb.predict(X_meta)
        meta_X = np.column_stack((p_lgbm_meta, p_xgb_meta))

        # 3. Обучаем мета-модель (Level 1)
        # Линейная регрессия найдет веса: например, 0.7*LGBM + 0.3*XGB
        meta_model = LinearRegression()
        meta_model.fit(meta_X, y_meta)

        # 4. Финальное предсказание на тесте
        p_lgbm_test = m_lgbm.predict(X_test)
        p_xgb_test = m_xgb.predict(X_test)
        test_meta_X = np.column_stack((p_lgbm_test, p_xgb_test))

        final_predictions[q] = meta_model.predict(test_meta_X)

    print(f"-> Ансамбль готов! Время сборки: {(time.time() - start_time) / 60:.2f} мин.")

    # === РЕЗУЛЬТАТЫ ===
    preds_lower = final_predictions[0.05]
    preds_median = final_predictions[0.50]
    preds_upper = final_predictions[0.95]

    inside_corridor = (y_test >= preds_lower) & (y_test <= preds_upper)
    coverage = np.mean(inside_corridor) * 100
    width = np.mean(preds_upper - preds_lower)

    print(f"\n======================================")
    print(f"=== РЕЗУЛЬТАТЫ СТЕКИНГА (LGBM+XGB) ===")
    print(f"======================================")
    print(f"Фактическое покрытие:      {coverage:.2f}%")
    print(f"Средняя ширина коридора:   {width:.4f}")

    # === ВИЗУАЛИЗАЦИЯ ===
    plt.figure(figsize=(15, 7))
    sample_size = min(150, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)

    y_s = y_test.iloc[indices].values
    l_s, m_s, u_s = preds_lower[indices], preds_median[indices], preds_upper[indices]

    sort_idx = np.argsort(m_s)
    x = np.arange(sample_size)

    plt.fill_between(x, l_s[sort_idx], u_s[sort_idx], color='purple', alpha=0.2, label='Коридор Ансамбля (90%)')
    plt.plot(x, m_s[sort_idx], color='purple', linewidth=2, label='Медиана Ансамбля')
    plt.scatter(x, y_s[sort_idx], color='red', s=20, alpha=0.6, label='Фактическая энтропия')

    plt.title('Стекинг: LightGBM + XGBoost (Финальный Блендинг)', fontsize=14)
    plt.legend()
    plt.show()


# ==========================================
# 4. ЗАПУСК
# ==========================================
if __name__ == "__main__":
    print("Загрузка файлов...")
    train_df = pd.read_csv('million.csv', sep=';', decimal=',', on_bad_lines='skip')
    test_df = pd.read_csv('unique_rows.csv', sep=';', decimal=',', on_bad_lines='skip', header=None)

    X_train, y_train = prepare_sequential_data(train_df, FEATURE_COLUMNS, TARGET_COLUMN)
    X_test, y_test = prepare_sequential_data(test_df, FEATURE_COLUMNS, TARGET_COLUMN)

    train_stacking_ensemble(X_train, y_train, X_test, y_test)