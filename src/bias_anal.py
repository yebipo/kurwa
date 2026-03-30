import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Импортируем библиотеки, использованные при обучении
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

warnings.filterwarnings('ignore')

# --- НАСТРОЙКИ ---
INPUT_FILE = 'million.csv'
OUTPUT_FILE = 'bias_report.csv'
MODEL_FILENAME = "final_model_stable.joblib"  # <-- ВАШЕ ИМЯ ФАЙЛА

# Если файл на Google Диске:
# from google.colab import drive
# drive.mount('/content/drive')
# MODEL_FILENAME = "/content/drive/MyDrive/Entropy_Project_Ultimate/final_model_god_with_bar.joblib"

CSV_SEP = ';'
CSV_DECIMAL = ','
RANDOM_STATE = 42
EPSILON = 1e-5


# --- КЛАССЫ (ОБЯЗАТЕЛЬНО ДЛЯ ЗАГРУЗКИ) ---
# Мы копируем их определения, чтобы joblib узнал структуру

def to_logit(y):
    y_safe = np.clip(y, EPSILON, 1 - EPSILON)
    return np.log(y_safe / (1 - y_safe))


def from_logit(z):
    res = 1 / (1 + np.exp(-z))
    return np.clip(res, 0, 1)


class SmartPreprocessor:
    def __init__(self, n_clusters=20):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
        self.n_clusters = n_clusters

    # Для загрузки достаточно сигнатуры метода, так как сам объект уже обучен
    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)
        df['row_mean'] = df.mean(axis=1)
        df['row_std'] = df.std(axis=1)
        df['row_max'] = df.max(axis=1)
        dists = self.kmeans.transform(X_scaled)
        for i in range(self.n_clusters):
            df[f'dist_clust_{i}'] = dists[:, i]
        df['cluster_id'] = self.kmeans.predict(X_scaled)
        return df


class IsotonicCorrector:
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    def fit(self, y_pred, y_true): self.iso.fit(y_pred, y_true); return self

    def predict(self, y_pred): return self.iso.predict(y_pred)


class HybridModelGod:
    def __init__(self, preprocessor, p_lgbm_q, p_lgbm_m, p_xgb_q, p_xgb_m, p_cat_q, p_cat_m):
        self.preprocessor = preprocessor
        self.bias_corrector = IsotonicCorrector()
        self.model = None

    def predict_logit_raw(self, X_raw):
        X = self.preprocessor.transform(X_raw)
        meta_features = np.column_stack([m.predict(X) for name, m in self.base_models])
        return self.final_estimator.predict(meta_features)

    def predict(self, X_raw, apply_calibration=True):
        logits = self.predict_logit_raw(X_raw)
        preds = from_logit(logits)
        if apply_calibration: preds = self.bias_corrector.predict(preds)
        return preds


# --- MAIN ---
def main():
    print("=== РАСЧЕТ BIAS (HYBRID GOD MODEL) ===")

    # 1. Загрузка модели
    if not os.path.exists(MODEL_FILENAME):
        print(f"❌ Модель {MODEL_FILENAME} не найдена!")
        return

    print(f"🔄 Загрузка модели: {MODEL_FILENAME}...")
    try:
        model = joblib.load(MODEL_FILENAME)
        print("✅ Модель успешно загружена!")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    # 2. Чтение данных
    print(f"📂 Читаем файл: {INPUT_FILE} ...")
    try:
        # Читаем только нужные колонки для экономии памяти
        # Col 2,3,4,5 = Фичи. Col 6 = Таргет
        df = pd.read_csv(INPUT_FILE, sep=CSV_SEP, decimal=CSV_DECIMAL, header=None, skiprows=1)
    except Exception as e:
        print(f"❌ Ошибка чтения: {e}")
        return

    # Проверка структуры
    if df.shape[1] < 6:
        print("❌ В файле меньше 6 колонок! (Нужно 4 фичи + 1 таргет)")
        return

    # Подготовка данных
    work_df = pd.DataFrame()
    work_df['feature_1'] = df.iloc[:, 1]
    work_df['feature_2'] = df.iloc[:, 2]
    work_df['feature_3'] = df.iloc[:, 3]
    work_df['feature_4'] = df.iloc[:, 4]

    real_entropy = pd.to_numeric(df.iloc[:, 5], errors='coerce')

    # Фильтр пустых
    valid_idx = real_entropy.dropna().index
    work_df = work_df.loc[valid_idx]
    real_entropy = real_entropy.loc[valid_idx]

    print(f"   Строк для анализа: {len(work_df)}")

    # 3. Предсказание
    print("🧠 Вычисляем предсказания (это займет время)...")
    # Используем apply_calibration=True, так как в финальной модели есть IsotonicCorrector
    pred_entropy = model.predict(work_df, apply_calibration=True)

    # 4. Расчет Bias (Real - Pred)
    bias = real_entropy - pred_entropy

    # 5. Сохранение
    print(f"💾 Сохраняем результат в {OUTPUT_FILE} ...")

    # Создаем копию исходного DataFrame
    df_out = df.iloc[valid_idx].copy()

    # Добавляем новые колонки (как в оригинале: 7-я предсказание, 8-я байас)
    # Индексы в pandas с 0, поэтому 6 и 7
    df_out[6] = pred_entropy
    df_out[7] = bias

    df_out.to_csv(OUTPUT_FILE, sep=';', decimal=',', index=False, header=False)

    # 6. Статистика
    mean_bias = bias.mean()
    median_bias = bias.median()
    std_bias = bias.std()

    print("\n" + "=" * 40)
    print("📊 СТАТИСТИКА ОШИБОК (REAL - PRED)")
    print("=" * 40)
    print(f"Mean Bias:   {mean_bias:+.6f}")
    print(f"Median Bias: {median_bias:+.6f}")
    print(f"Std Dev:     {std_bias:.6f}")

    # Анализ знака
    if mean_bias > 0:
        print("\n📉 ВЫВОД: Модель ЗАНИЖАЕТ (Осторожная)")
        print(f"   В среднем реальность выше прогноза на {abs(mean_bias):.6f}")
    else:
        print("\n📈 ВЫВОД: Модель ЗАВЫШАЕТ (Оптимистичная)")
        print(f"   В среднем реальность ниже прогноза на {abs(mean_bias):.6f}")

    # Анализ высоких значений (>0.9)
    high_mask = real_entropy > 0.90
    if high_mask.sum() > 0:
        high_bias = bias[high_mask].mean()
        print("\n🎯 НА ВЫСОКИХ ЗНАЧЕНИЯХ (>0.90):")
        print(f"   Mean Bias: {high_bias:+.6f}")
        if abs(high_bias) < 0.01:
            print("   ✅ Отличная точность на пиках!")
        else:
            print("   ⚠️ Есть смещение на пиках.")

    # 7. График
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(bias, bins=100, kde=True, color='teal')
    plt.axvline(0, color='black', linestyle='-')
    plt.axvline(mean_bias, color='red', linestyle='--', label=f'Mean: {mean_bias:.4f}')
    plt.title('Гистограмма Ошибок (Bias)')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(real_entropy, bias, alpha=0.1, s=1, c='purple')
    plt.axhline(0, color='black')
    plt.title('Ошибка vs Реальное значение')
    plt.xlabel('Real Entropy')
    plt.ylabel('Bias')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()