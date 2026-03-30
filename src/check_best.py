import pandas as pd
import numpy as np
import joblib
import os

# --- НАСТРОЙКИ ---
# Имя файла модели. Если вы переименовали его в final_model.joblib, поменяйте тут.
# По умолчанию мы создавали "production_model_v26.pkl"
MODEL_FILENAME = "final_model.joblib"

# Найденный идеал
BEST_F1 = 1
BEST_F2 = 3
BEST_F3 = 4194304
BEST_F4_CENTER = 16609695

# Поправка на пессимизм модели (которую мы вычислили ранее)
BIAS = 0.048147

# --- КЛАССЫ (ОБЯЗАТЕЛЬНО) ---
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


class GlobalPreprocessor:
    def __init__(self, scaler, kmeans):
        self.scaler = scaler
        self.kmeans = kmeans

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        cluster_labels = self.kmeans.predict(X_scaled)
        X_out = X_log.copy()
        X_out['cluster_id'] = cluster_labels
        return X_out


class ProductionModel:
    def __init__(self, preprocessor, lgbm_p, xgb_p, cat_p):
        self.preprocessor = preprocessor
        self.estimators = [
            ('lgbm', lgb.LGBMRegressor(**lgbm_p)),
            ('xgb', xgb.XGBRegressor(**xgb_p)),
            ('catboost', cb.CatBoostRegressor(**cat_p))
        ]
        self.model = StackingRegressor(estimators=self.estimators, final_estimator=LinearRegression(), cv=3, n_jobs=1)

    def fit(self, X_raw, y): pass

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        return self.model.predict(X_ready)


def main():
    print("=== ПРОГОН WINNING PARAMS ЧЕРЕЗ МОДЕЛЬ ===")

    # Загрузка (с проверкой имени)
    fname = MODEL_FILENAME
    if not os.path.exists(fname):
        # Проверка альтернативного имени, если вы переименовали
        if os.path.exists("final_model.joblib"):
            fname = "final_model.joblib"
        else:
            print(f"❌ Файл модели {fname} не найден.")
            return

    print(f"⏳ Загрузка {fname}...")
    model = joblib.load(fname)
    print("✅ Модель в памяти.")

    # 1. Генерируем 20 точек вокруг центра (±5%)
    # Чтобы увидеть, где именно пик
    min_f4 = BEST_F4_CENTER * 0.95
    max_f4 = BEST_F4_CENTER * 1.05
    f4_values = np.linspace(min_f4, max_f4, 20)

    df = pd.DataFrame({
        'feature_1': BEST_F1,
        'feature_2': BEST_F2,
        'feature_3': BEST_F3,
        'feature_4': f4_values
    })
    df['feature_4'] = df['feature_4'].astype(int)  # Округляем до целых

    # 2. Предсказываем
    print("🧠 Вычисление...")
    preds = model.predict(df)

    df['Predicted'] = preds
    df['Expected_Real'] = preds + BIAS  # Добавляем поправку

    # 3. Сортируем и показываем
    df = df.sort_values(by='Predicted', ascending=False)

    print("\n🏆 РЕЗУЛЬТАТЫ ПРОГОНА (ТОП-10):")
    print(f"{'F1':<3} | {'F2':<3} | {'F3':<10} | {'F4 (Variable)':<15} | {'Model Pred':<10} | {'REAL (Est)':<10}")
    print("-" * 70)

    for _, row in df.head(10).iterrows():
        print(
            f"{int(row['feature_1']):<3} | {int(row['feature_2']):<3} | {int(row['feature_3']):<10} | {int(row['feature_4']):<15_} | {row['Predicted']:.6f}   | {row['Expected_Real']:.6f}")

    best_row = df.iloc[0]
    print("\n🔥 ЛУЧШАЯ КОМБИНАЦИЯ:")
    print(f"   F1: {int(best_row['feature_1'])}")
    print(f"   F2: {int(best_row['feature_2'])}  <-- (Секретный ингредиент)")
    print(f"   F3: {int(best_row['feature_3'])}")
    print(f"   F4: {int(best_row['feature_4'])}")
    print(f"   Прогноз модели: {best_row['Predicted']:.6f}")
    print(f"   Ожидаемая реальность: {best_row['Expected_Real']:.6f}")


if __name__ == "__main__":
    main()