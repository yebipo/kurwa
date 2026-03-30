import pandas as pd
import numpy as np
import joblib
import os
import warnings

# Импортируем все библиотеки, которые использовались при обучении
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
# Укажите путь к вашему файлу модели (который скачали с Диска)
MODEL_FILENAME = "final_model_stable.joblib"
# Если модель лежит на диске, раскомментируйте и укажите путь:
# from google.colab import drive
# drive.mount('/content/drive')
# MODEL_FILENAME = "/content/drive/MyDrive/Entropy_Project_Ultimate/final_model_god_20_clusters.joblib"

RANDOM_STATE = 42
EPSILON = 1e-5

# 1. ВАШ ЛУЧШИЙ РЕЗУЛЬТАТ (ИЗ EXCEL/ИСТОРИИ)
# Введите сюда параметры, которые дали вам максимальную энтропию
USER_BEST_PARAMS = {
    'feature_1': [1],
    'feature_2': [0],
    'feature_3': [2097152],
    'feature_4': [1169894]
}
USER_REAL_ENTROPY = 0.954702  # Реальное значение для этих параметров

# 2. ТОЧКА, КОТОРУЮ ПРЕДЛАГАЕТ АЛГОРИТМ (ДЛЯ СРАВНЕНИЯ)
ALGO_BEST_PARAMS = {
    'feature_1': [1],
    'feature_2': [3],
    'feature_3': [4194304],
    'feature_4': [16609695]
}


# ALGO_PRED_ENTROPY мы сейчас вычислим сами моделью

# --- НЕОБХОДИМЫЕ КЛАССЫ (Чтобы joblib понял структуру) ---
# Их нужно просто объявить, код внутри не важен, главное - наличие классов.

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

    # Методы fit/transform внутри joblib уже сохранены,
    # здесь их переписывать полностью не обязательно, но для корректности оставим структуру
    def transform(self, X_raw):
        # Логика воспроизведения трансформации
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
        self.model = None  # Заглушка, реальная модель загрузится из файла

    def predict_logit_raw(self, X_raw):
        # Метод для получения сырых логитов (до сигмоиды)
        X = self.preprocessor.transform(X_raw)
        # В joblib объекте self.base_models и self.final_estimator уже есть
        meta_features = np.column_stack([m.predict(X) for name, m in self.base_models])
        return self.final_estimator.predict(meta_features)

    def predict(self, X_raw, apply_calibration=True):
        logits = self.predict_logit_raw(X_raw)
        preds = from_logit(logits)
        if apply_calibration: preds = self.bias_corrector.predict(preds)
        return preds


# --- ЗАПУСК ---
def main():
    print("=== АНАЛИЗ ГИПОТЕЗЫ (GOD MODEL) ===")

    if not os.path.exists(MODEL_FILENAME):
        print(f"❌ Файл {MODEL_FILENAME} не найден! Загрузите его или укажите верный путь.")
        return

    print(f"🔄 Загрузка модели: {MODEL_FILENAME}...")
    try:
        model = joblib.load(MODEL_FILENAME)
        print("✅ Модель успешно загружена!")
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    # --- 1. ПРОВЕРКА ВАШЕЙ ТОЧКИ ---
    df_user = pd.DataFrame(USER_BEST_PARAMS)

    # Предсказываем
    pred_user = model.predict(df_user, apply_calibration=True)[0]

    print(f"\n1️⃣ ВАШ ЛУЧШИЙ РЕЗУЛЬТАТ:")
    print(f"   Параметры: {USER_BEST_PARAMS}")
    print(f"   РЕАЛЬНАЯ Энтропия: {USER_REAL_ENTROPY:.6f}")
    print(f"   ПРОГНОЗ Модели:    {pred_user:.6f}")

    # Считаем Байас (насколько модель врет в этой зоне)
    bias = USER_REAL_ENTROPY - pred_user
    if bias > 0:
        print(f"   📉 Модель ЗАНИЖАЕТ на: {bias:.6f} (Осторожная)")
    else:
        print(f"   📈 Модель ЗАВЫШАЕТ на: {abs(bias):.6f} (Оптимистичная)")

    # --- 2. ПРОВЕРКА ТОЧКИ АЛГОРИТМА ---
    df_algo = pd.DataFrame(ALGO_BEST_PARAMS)
    pred_algo = model.predict(df_algo, apply_calibration=True)[0]

    print(f"\n2️⃣ ПРЕДЛОЖЕНИЕ АЛГОРИТМА:")
    print(f"   Параметры: {ALGO_BEST_PARAMS}")
    print(f"   ПРОГНОЗ Модели:    {pred_algo:.6f}")

    # --- 3. ВЕРДИКТ ---
    # Корректируем прогноз алгоритма на найденный байас
    estimated_real_algo = pred_algo + bias

    print(f"\n🚀 ИТОГОВЫЙ АНАЛИЗ:")
    print(f"   Если перенести ошибку модели ({bias:+.6f}) на новую точку...")
    print(f"   ОЖИДАЕМАЯ РЕАЛЬНОСТЬ: ~{estimated_real_algo:.6f}")
    print("-" * 30)

    if estimated_real_algo > USER_REAL_ENTROPY:
        diff = estimated_real_algo - USER_REAL_ENTROPY
        print(f"🎉 ПОБЕДА! Алгоритм, вероятно, нашел точку ЛУЧШЕ на +{diff:.6f}")
        print("   Рекомендуется проверить эти параметры на практике!")
    elif estimated_real_algo > (USER_REAL_ENTROPY - 0.005):
        print("🤔 СПОРНО. Результат очень близок к вашему. Возможно, это плато.")
    else:
        print("❄️ ХОЛОДНО. Алгоритм нашел локальный максимум, но он хуже вашего рекорда.")


if __name__ == "__main__":
    main()