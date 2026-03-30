# --- 0. ИМПОРТЫ ---
# !pip install optuna catboost xgboost lightgbm scikit-learn pandas numpy matplotlib seaborn

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.cluster import KMeans
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.isotonic import IsotonicRegression
import xgboost as xgb
import catboost as cb
import warnings

warnings.filterwarnings('ignore')

# --- 1. НАСТРОЙКИ ---

MODEL_FILENAME = "final_model_stable.joblib"
INPUT_FILE = 'million.csv'
GATEKEEPER_FILENAME = "gatekeeper_classifier.joblib"

RANDOM_STATE = 42
# Порог: Всё, что ниже 0.85 в реальности, считаем "неинтересным"
HIGH_VALUE_THRESHOLD = 0.9
# Значение, которое мы ставим для отсеянного мусора (среднее по низу)
LOW_VALUE_FILL = 0.20


# --- 2. КЛАССЫ (ОБЯЗАТЕЛЬНО ДЛЯ ЗАГРУЗКИ JOBLIB) ---
# Эти классы должны быть определены точь-в-точь как при обучении

# --- 2. КЛАССЫ (ПОЛНЫЕ ОПРЕДЕЛЕНИЯ!) ---

def to_logit(y):
    y_safe = np.clip(y, 1e-5, 1 - 1e-5)
    return np.log(y_safe / (1 - y_safe))


def from_logit(z):
    res = 1 / (1 + np.exp(-z))
    return np.clip(res, 0, 1)


class SmartPreprocessor:
    def __init__(self, n_clusters=20):
        # Инициализация для joblib
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init='auto')
        self.n_clusters = n_clusters

    def fit(self, X):
        # Fit не нужен при предсказании, но заглушку оставить можно
        pass

    def transform(self, X_raw):
        # !!! ЗДЕСЬ ДОЛЖНА БЫТЬ ПОЛНАЯ ЛОГИКА, А НЕ PASS !!!

        # 1. Базовая трансформация
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        # self.scaler уже обучен и загружен из файла, используем его
        X_scaled = self.scaler.transform(X_log)
        df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)

        # 2. Статистика
        df['row_mean'] = df.mean(axis=1)
        df['row_std'] = df.std(axis=1)
        df['row_max'] = df.max(axis=1)

        # 3. Кластеры (self.kmeans тоже загружен из файла)
        dists = self.kmeans.transform(X_scaled)
        for i in range(self.n_clusters):
            df[f'dist_clust_{i}'] = dists[:, i]
        df['cluster_id'] = self.kmeans.predict(X_scaled)

        return df


class IsotonicCorrector:
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    def fit(self, y_p, y_t): pass

    def predict(self, y_p): return self.iso.predict(y_p)


class HybridModelGod:
    def __init__(self, preprocessor, p_lgbm_q, p_lgbm_m, p_xgb_q, p_xgb_m, p_cat_q, p_cat_m):
        self.preprocessor = preprocessor
        self.bias_corrector = IsotonicCorrector()
        self.model = None
        self.base_models = []  # Будет заполнено joblib
        self.final_estimator = None  # Будет заполнено joblib

    def predict_logit_raw(self, X_raw):
        # Препроцессинг
        X = self.preprocessor.transform(X_raw)
        # Предсказания базовых моделей
        meta_features = np.column_stack([m.predict(X) for name, m in self.base_models])
        # Предсказание мета-модели
        return self.final_estimator.predict(meta_features)

    def predict(self, X_raw, apply_calibration=True):
        logits = self.predict_logit_raw(X_raw)
        preds = from_logit(logits)
        if apply_calibration: preds = self.bias_corrector.predict(preds)
        return preds

# --- 3. MAIN (КАСКАД + ГРАФИКИ) ---
def main():
    print("\n=== СОЗДАНИЕ И АНАЛИЗ КАСКАДА (ФЕЙС-КОНТРОЛЬ) ===\n")

    # 1. Загрузка данных
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Файл {INPUT_FILE} не найден.")
        return

    print("📂 Читаем данные...")
    df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, skiprows=1,
                     usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)

    X = df.drop(columns=['target'])
    y_reg = df['target']

    # 2. Подготовка Классификации
    # Цель: 1 если энтропия > 0.85, иначе 0
    y_class = (y_reg > HIGH_VALUE_THRESHOLD).astype(int)

    print(f"🎯 Разметка для классификатора (Порог {HIGH_VALUE_THRESHOLD}):")
    print(f"   Класс 1 (Важные): {y_class.sum()} строк")
    print(f"   Класс 0 (Мусор):  {len(y_class) - y_class.sum()} строк")

    # Сплит
    X_train, X_test, y_train_clf, y_test_clf = train_test_split(X, y_class, test_size=0.2, random_state=RANDOM_STATE)
    # Нам также нужны реальные значения регрессии для теста, чтобы проверить байас
    y_test_reg = y_reg.loc[X_test.index]

    # 3. Обучение Gatekeeper
    print("\n🚧 Обучение Gatekeeper (LightGBM Classifier)...")
    clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=40,
        class_weight='balanced',  # Чтобы не пропустить редкие алмазы
        n_jobs=-1,
        verbose=-1,
        random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train_clf)

    probs = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test_clf, probs)
    print(f"✅ Gatekeeper готов. ROC AUC: {auc:.4f}")

    # 4. Загрузка Бога
    print("\n🔄 Загрузка God Model (Регрессор)...")
    try:
        god_model = joblib.load(MODEL_FILENAME)
        print("   Успешно.")
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return

    # 5. КАСКАДНОЕ ПРЕДСКАЗАНИЕ
    print("\n🚀 Запуск Каскада на тестовой выборке...")

    # Шаг А: Фейс-контроль
    # Если вероятность быть топом > 0.4 (берем с запасом), пускаем дальше
    gate_mask = probs > 0.4

    print(f"   Отправлено к Богу: {gate_mask.sum()} из {len(X_test)} ({gate_mask.mean() * 100:.1f}%)")

    # Шаг Б: Регрессия для избранных
    final_preds = np.full(len(X_test), LOW_VALUE_FILL)  # Сначала всем ставим 0.2

    if gate_mask.sum() > 0:
        X_passed = X_test[gate_mask]
        # Тяжелая артиллерия
        preds_god = god_model.predict(X_passed, apply_calibration=True)
        final_preds[gate_mask] = preds_god

    # 6. АНАЛИЗ РЕЗУЛЬТАТОВ
    bias = y_test_reg - final_preds
    rmse = np.sqrt(mean_squared_error(y_test_reg, final_preds))
    r2 = r2_score(y_test_reg, final_preds)

    print("-" * 40)
    print(f"📊 ИТОГИ КАСКАДА:")
    print(f"   RMSE: {rmse:.5f}")
    print(f"   R2:   {r2:.5f}")
    print(f"   Mean Bias: {bias.mean():.5f} (Был -0.10, стал ближе к 0?)")
    print("-" * 40)

    # Сохраняем классификатор
    joblib.dump(clf, GATEKEEPER_FILENAME)
    print(f"💾 Классификатор сохранен: {GATEKEEPER_FILENAME}")

    # 7. ГРАФИКИ (Bias Analysis)
    print("\n🎨 Рисуем графики...")
    plt.figure(figsize=(18, 5))

    # График 1: Real vs Pred
    plt.subplot(1, 3, 1)
    plt.scatter(y_test_reg, final_preds, alpha=0.1, s=3, c='blue')
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.title(f"Real vs Cascade Prediction\n(R2: {r2:.3f})")
    plt.xlabel("Real Entropy")
    plt.ylabel("Cascade Prediction")
    plt.grid(True, alpha=0.3)

    # График 2: Гистограмма Байаса
    plt.subplot(1, 3, 2)
    sns.histplot(bias, bins=100, kde=True, color='purple')
    plt.axvline(0, color='red', linestyle='--', lw=2)
    plt.title(f"Распределение Ошибок (Bias)\nMean: {bias.mean():.4f}")
    plt.xlabel("Error (Real - Pred)")

    # График 3: Ошибка от Значения (Проверка хвостов)
    plt.subplot(1, 3, 3)
    plt.scatter(y_test_reg, bias, alpha=0.1, s=3, c='green')
    plt.axhline(0, color='red', linestyle='--', lw=2)
    plt.title("Где ошибаемся?\n(Error vs Real Value)")
    plt.xlabel("Real Entropy")
    plt.ylabel("Error")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()