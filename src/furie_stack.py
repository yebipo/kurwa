# ==========================================
# 0. ИМПОРТ БИБЛИОТЕК
# ==========================================
import pandas as pd
import numpy as np
import time
import gc
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from numpy.lib.stride_tricks import sliding_window_view

sns.set_theme(style="whitegrid")

# ==========================================
# 1. НАСТРОЙКИ И ПАРАМЕТРЫ
# ==========================================
FEATURE_COLUMNS = [1, 2, 3, 4]
TARGET_COLUMN = 5

# Наши лучшие "спартанские" настройки
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
# 2. ФУНКЦИИ ОБРАБОТКИ (MEMORY & NAMES OPTIMIZED)
# ==========================================

def add_fft_features_safe(df, window=64, n_components=5, chunk_size=100000):
    """Потоковый расчет FFT для экономии RAM"""
    print(f"-> Вычисляем FFT (окно {window}, кусками по {chunk_size})...")
    vals = df['target'].values.astype(np.float32)
    n_rows = len(df)
    fft_feats = np.zeros((n_rows, n_components), dtype=np.float32)

    start_idx = window - 1
    for i in range(start_idx, n_rows, chunk_size):
        end = min(i + chunk_size, n_rows)
        chunk_data_start = i - window + 1
        chunk_data = vals[chunk_data_start:end]

        # Создаем скользящие окна
        windows = sliding_window_view(chunk_data, window_shape=window)
        # Центрируем сигнал и считаем Real FFT
        windows_centered = windows - windows.mean(axis=1)[:, None]
        f_transform = np.abs(np.fft.rfft(windows_centered, axis=1))

        # Записываем компоненты (пропуская 0-ю, которая отвечает за среднее)
        fft_feats[i:end, :] = f_transform[:, 1:n_components + 1]

        del windows, windows_centered, f_transform
        gc.collect()

    fft_cols = [f'fft_{i}' for i in range(n_components)]
    fft_df = pd.DataFrame(fft_feats, columns=fft_cols, index=df.index)
    return pd.concat([df, fft_df], axis=1)


def prepare_data_ultra_safe(filepath, is_test=False):
    """Загрузка с жестким переименованием колонок для XGBoost"""
    print(f"-> Загрузка {filepath}...")
    cols_to_read = FEATURE_COLUMNS + [TARGET_COLUMN]
    header = None if is_test else 'infer'

    df = pd.read_csv(filepath, sep=';', decimal=',', on_bad_lines='skip',
                     usecols=cols_to_read, header=header, engine='c')

    # 🔥 ИСПРАВЛЕНИЕ FEATURE NAMES MISMATCH 🔥
    # Принудительно называем колонки одинаково для всех файлов
    new_names = [f'f{i + 1}' for i in range(len(FEATURE_COLUMNS))] + ['target']
    df.columns = new_names

    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', '.').str.strip(), errors='coerce')
        df[col] = df[col].astype(np.float32)

    y = df['target'].copy()
    X = df.drop(columns=['target'])
    del df
    gc.collect()

    # Генерация временных фич
    X['prev_1'] = y.shift(1)
    X['roll_mean'] = y.rolling(window=10).mean().shift(1)
    X['roll_std'] = y.rolling(window=10).std().shift(1)

    # Добавление FFT
    temp_df = pd.concat([X, y.rename('target')], axis=1)
    temp_df = add_fft_features_safe(temp_df, window=64, n_components=5)

    # Фильтрация пустых строк
    valid_idx = temp_df['fft_0'].notna() & (temp_df['fft_0'] != 0)
    final_df = temp_df[valid_idx].copy()
    final_target = final_df.pop('target')

    del temp_df
    gc.collect()
    return final_df, final_target


# ==========================================
# 3. КВАНТИЛЬНЫЙ СТЕКЕР (LEVEL 1)
# ==========================================

class QuantileStacker:
    def __init__(self, alpha):
        self.alpha = alpha
        self.lgbm = LGBMRegressor(objective='quantile', alpha=alpha, **LGBM_PARAMS)
        self.xgb = XGBRegressor(objective='reg:quantileerror', quantile_alpha=alpha, **XGB_PARAMS)
        # Мета-модель тоже квантильная, чтобы не схлопывать коридор
        self.meta = LGBMRegressor(objective='quantile', alpha=alpha, n_estimators=50, max_depth=3, verbose=-1)

    def fit(self, X, y):
        split = int(len(X) * 0.8)
        X_l0, X_meta = X.iloc[:split], X.iloc[split:]
        y_l0, y_meta = y.iloc[:split], y.iloc[split:]

        print(f"      [L0] Обучение базовых моделей для q={self.alpha}...")
        self.lgbm.fit(X_l0, y_l0)
        self.xgb.fit(X_l0, y_l0)

        print(f"      [L1] Обучение мета-модели...")
        m_feats = np.column_stack([self.lgbm.predict(X_meta), self.xgb.predict(X_meta)])
        self.meta.fit(m_feats, y_meta)
        return self

    def predict(self, X):
        m_feats = np.column_stack([self.lgbm.predict(X), self.xgb.predict(X)])
        return self.meta.predict(m_feats)


# ==========================================
# 4. ФИНАЛЬНЫЙ ЗАПУСК И ЗАМЕРЫ
# ==========================================

if __name__ == "__main__":
    start_time_all = time.time()
    try:
        # Загрузка и подготовка
        X_train, y_train = prepare_data_ultra_safe('million.csv', is_test=False)
        X_test, y_test = prepare_data_ultra_safe('unique_rows.csv', is_test=True)

        print(f"\nДанные в порядке. Фичей: {X_train.shape[1]}")

        # Обучение ансамблей
        stacker_lower = QuantileStacker(alpha=0.05).fit(X_train, y_train)
        stacker_upper = QuantileStacker(alpha=0.95).fit(X_train, y_train)

        # Предсказание на unique_rows.csv
        print("\n-> Финальное тестирование на unique_rows.csv...")
        p_low = stacker_lower.predict(X_test)
        p_up = stacker_upper.predict(X_test)

        # Метрики
        cov = np.mean((y_test >= p_low) & (y_test <= p_up)) * 100
        w = np.mean(p_up - p_low)

        print(f"\n" + "=" * 40)
        print(f"   РЕЗУЛЬТАТЫ СПЕКТРАЛЬНОГО СТЕКИНГА")
        print(f"   Фактическое покрытие: {cov:.2f}%")
        print(f"   Средняя ширина:       {w:.4f}")
        print(f"   Время выполнения:     {(time.time() - start_time_all) / 60:.2f} мин.")
        print("=" * 40)

        # Визуализация 100 точек
        plt.figure(figsize=(12, 6))
        plt.fill_between(range(100), p_low[:100], p_up[:100], color='orange', alpha=0.3, label='FFT-Stacking Corridor')
        plt.scatter(range(100), y_test[:100], color='red', s=15, label='Actual Entropy')
        plt.title(f"FFT + Quantile Stacking на Unique Rows (Coverage: {cov:.2f}%)")
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"\n[!] ОШИБКА: {e}")
        import traceback

        traceback.print_exc()