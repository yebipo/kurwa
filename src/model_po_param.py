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
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor
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
# 3. БАЗОВЫЕ МОДЕЛИ (LightGBM + XGBoost)
# ==========================================
lgbm_model = LGBMRegressor(
    reg_alpha=2.6508280629175112e-08,
    reg_lambda=0.009910745904773404,
    learning_rate=0.029914617989631492,
    max_depth=24,
    min_child_samples=14,
    n_estimators=1037,
    num_leaves=192,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

xgb_model = XGBRegressor(
    colsample_bytree=0.9394261401745961,
    learning_rate=0.036028372449179814,
    max_depth=11,
    n_estimators=790,
    subsample=0.9820423568547401,
    random_state=42,
    tree_method='hist',
    device='cuda'
)

# ==========================================
# 4. СБОРКА АНСАМБЛЯ
# ==========================================
stacking_model = StackingRegressor(
    estimators=[
        ('lgbm', lgbm_model),
        ('xgb', xgb_model)
    ],
    final_estimator=Ridge(),
    n_jobs=1,
    passthrough=False
)


# ==========================================
# 5. ОСНОВНОЙ ПАЙПЛАЙН
# ==========================================
def train_evaluate_and_plot(X_train, y_train, X_test, y_test):
    print("-> 1. Генерация признака cluster_id (KMeans)...")
    cluster_extractor = ClusterFeatureExtractor(n_clusters=4)
    X_train_eng = cluster_extractor.fit_transform(X_train)
    X_test_eng = cluster_extractor.transform(X_test)

    print("-> 2. Обучение ансамбля (ожидайте ~5-15 минут)...")
    start_time = time.time()
    stacking_model.fit(X_train_eng, y_train)
    print(f"-> Обучение завершено за {(time.time() - start_time) / 60:.2f} минут!")

    print("-> 3. Предсказание на тесте...")
    preds = stacking_model.predict(X_test_eng)

    # Отбрасываем NaN значения, если битая строка проскочила
    valid_idx = ~np.isnan(y_test) & ~np.isnan(preds)
    y_test_clean = y_test[valid_idx]
    preds_clean = preds[valid_idx]

    rmse = np.sqrt(mean_squared_error(y_test_clean, preds_clean))
    mae = mean_absolute_error(y_test_clean, preds_clean)
    r2 = r2_score(y_test_clean, preds_clean)
    mape = np.mean(np.abs((y_test_clean - preds_clean) / y_test_clean)) * 100

    print(f"\n=======================")
    print(f"=== РЕЗУЛЬТАТЫ TEST ===")
    print(f"=======================")
    print(f"R²:   {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"MAPE: {mape:.2f}%\n")

    joblib.dump(cluster_extractor, 'kmeans_clusterer.joblib')
    joblib.dump(stacking_model, 'stacking_model.joblib')

    print("-> 4. Отрисовка графиков...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.scatterplot(x=y_test_clean, y=preds_clean, alpha=0.3, ax=axes[0], color='blue')
    min_val, max_val = min(y_test_clean.min(), preds_clean.min()), max(y_test_clean.max(), preds_clean.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    axes[0].set_title('Факт vs Предсказание')

    residuals = y_test_clean - preds_clean
    sns.histplot(residuals, kde=True, ax=axes[1], color='purple')
    axes[1].set_title('Распределение ошибок')

    model_names = ['LightGBM', 'XGBoost']
    meta_weights = stacking_model.final_estimator_.coef_
    sns.barplot(x=model_names, y=meta_weights, ax=axes[2], palette='viridis')
    axes[2].set_title('Веса моделей (Ridge)')

    plt.tight_layout()
    plt.show()


# ==========================================
# 6. ЗАПУСК
# ==========================================
if __name__ == "__main__":
    print("Загрузка данных...")

    train_df = pd.read_csv('million.csv', sep=';', decimal=',', on_bad_lines='skip')
    # Лекарство для теста: header=None!
    test_df = pd.read_csv('unique_rows.csv', sep=';', decimal=',', on_bad_lines='skip', header=None)

    print(f"\nУспешно загружено (чистых строк):")
    print(f"Train: {train_df.shape[0]} строк")
    print(f"Test:  {test_df.shape[0]} строк\n")

    X_train = train_df.iloc[:, FEATURE_COLUMNS]
    y_train = train_df.iloc[:, TARGET_COLUMN]

    X_test = test_df.iloc[:, FEATURE_COLUMNS]
    y_test = test_df.iloc[:, TARGET_COLUMN]

    # Унифицируем имена колонок, чтобы модели не ругались
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_train.columns  # Жестко копируем имена из трейна в тест

    train_evaluate_and_plot(X_train, y_train, X_test, y_test)