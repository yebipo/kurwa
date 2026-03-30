import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.compose import TransformedTargetRegressor
import lightgbm as lgb
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
INPUT_FILE = 'unique_rows.csv'
MODEL_NAME = "final_model_v2_logit.joblib"
RANDOM_STATE = 42


# 1. ТРАНСФОРМАЦИЯ ТАРГЕТА (Логит-функции)
def logit_func(x):
    x = np.clip(x, 1e-7, 1 - 1e-7)
    return np.log(x / (1 - x))


def expit_func(x):
    return 1 / (1 + np.exp(-x))


# 2. ПРОДВИНУТЫЙ ПРЕПРОЦЕССОР
class AdvancedPreprocessor:
    def __init__(self):
        # Превращаем любое распределение в "колокол" (нормальное)
        self.qt = QuantileTransformer(output_distribution='normal', random_state=RANDOM_STATE)

    def fit(self, X):
        X_clean = X.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
        # log1p помогает сжать гигантские значения признаков перед квантилем
        self.qt.fit(np.log1p(X_clean.clip(lower=0)))
        return self

    def transform(self, X):
        X_clean = X.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_log = np.log1p(X_clean.clip(lower=0))
        X_qt = self.qt.transform(X_log)
        return pd.DataFrame(X_qt, columns=X.columns)


# 3. ОСНОВНОЙ СКРИПТ
def main():
    print("🚀 Старт обучения модели v2 (Logit Target + Weights)")

    # Загрузка и чистка
    try:
        df = pd.read_csv(INPUT_FILE, sep=';', decimal=',', header=None, on_bad_lines='skip', engine='python')
        df = df[[1, 2, 3, 4, 5]]
        df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
    except Exception as e:
        print(f"❌ Ошибка загрузки: {e}")
        return

    X = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']]
    y = df['target'].values

    # Сплит
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # РАСЧЕТ ВЕСОВ: Чем реже значение таргета, тем важнее эта строка для модели
    # Это убивает "пофигизм" модели к редким зонам
    y_counts, bins = np.histogram(y_train, bins=20)
    bin_indices = np.digitize(y_train, bins[1:-1])
    weights = 1.0 / (y_counts[bin_indices] + 1)
    weights = weights / np.mean(weights)

    # Подготовка данных
    pre = AdvancedPreprocessor()
    X_train_pre = pre.fit(X_train).transform(X_train)
    X_test_pre = pre.transform(X_test)

    # Модели (без CatBoost, как и просил)
    lgbm = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        objective='regression',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1
    )

    xgboost = xgb.XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        tree_method='hist',  # Для скорости
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    # Ансамбль с параллельным обучением
    stack = StackingRegressor(
        estimators=[('lgbm', lgbm), ('xgb', xgboost)],
        final_estimator=Ridge(alpha=0.5),
        cv=3,  # Ускоряем обучение
        n_jobs=-1
    )

    # Обертка для работы в logit-пространстве
    final_model = TransformedTargetRegressor(
        regressor=stack,
        func=logit_func,
        inverse_func=expit_func
    )

    print("🧠 Обучаем (время на чай: 20-30 минут)...")
    # Передаем веса только в regressor часть (stacking)
    final_model.fit(X_train_pre, y_train, regressor__sample_weight=weights)

    # Предикт и метрики
    y_pred = final_model.predict(X_test_pre)
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    print("\n" + "=" * 45)
    print(f"✅ Финальный RMSE: {rmse:.6f}")
    print("=" * 45)

    # Сохранение
    joblib.dump({'model': final_model, 'preprocessor': pre}, MODEL_NAME)
    print(f"💾 Модель сохранена как {MODEL_NAME}")

    # ГРАФИКИ: Проверка на "веер"
    plt.figure(figsize=(16, 7))

    # 1. Диагональ
    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.1, s=2, color='darkblue')
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.title("Real vs Predicted (Logit Target)")

    # 2. Остатки (Residuals) - ищем гетероскедастичность
    plt.subplot(1, 2, 2)
    residuals = y_pred - y_test
    plt.scatter(y_pred, residuals, alpha=0.1, s=2, color='crimson')
    plt.axhline(0, color='black', linestyle='--')
    plt.title("Residuals Plot (Checking for Symmetry)")
    plt.xlabel("Predictions")
    plt.ylabel("Error")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()