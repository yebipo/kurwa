# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# --- ВАЖНО: Определения классов должны совпадать с исходными для joblib ---
class SmartPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=15):
        self.scaler = QuantileTransformer(output_distribution='normal', random_state=42)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        self.n_clusters = n_clusters

    def fit(self, X_raw, y=None): return self

    def transform(self, X_raw):
        X_log = np.log2(X_raw.astype(float).clip(lower=1))
        X_scaled = self.scaler.transform(X_log)
        df = pd.DataFrame(X_scaled, columns=X_raw.columns, index=X_raw.index)
        df['row_mean'] = df.mean(axis=1)
        df['row_std'] = df.std(axis=1)
        dists = self.kmeans.transform(X_scaled)
        for i in range(self.n_clusters):
            df[f'dist_clust_{i}'] = dists[:, i]
        df['cluster_id'] = self.kmeans.predict(X_scaled)
        return df


class IsotonicCorrector:
    def __init__(self):
        self.iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')

    def fit(self, y_pred, y_true):
        self.iso.fit(y_pred, y_true)
        return self

    def predict(self, y_pred):
        return self.iso.predict(y_pred)


class HybridModelGod:
    def __init__(self, p_lgbm_q, p_lgbm_m, p_xgb_q, p_xgb_m, p_cat_q, p_cat_m):
        self.bias_corrector = IsotonicCorrector()
        self.final_preprocessor = None
        self.final_estimator = Ridge(alpha=1.0, random_state=42)
        self.trained_models = []

    def predict(self, X_raw, calibrate=True):
        X = self.final_preprocessor.transform(X_raw)
        preds = []
        for m in self.trained_models:
            preds.append(m.predict(X))
        meta = np.column_stack(preds)
        logits = self.final_estimator.predict(meta)
        res = 1 / (1 + np.exp(-logits))
        res = np.clip(res, 0, 1)
        if calibrate: res = self.bias_corrector.predict(res)
        return res

    def calibrate(self, X_calib, y_calib):
        y_pred_unc = self.predict(X_calib, calibrate=False)
        self.bias_corrector.fit(y_pred_unc, y_calib)


# --- НОВЫЙ КЛАСС: Блендинг вместо Ridge ---
class AverageBlender(BaseEstimator, RegressorMixin):
    def fit(self, X, y): return self

    def predict(self, X):
        return np.mean(X, axis=1)


# --- КОНФИГ ---
MODEL_PATH = 'final_model_god_v4.joblib'
DATA_FILE = 'million.csv'
NEW_MODEL_NAME = 'final_model_god_v4_OPTIMIZED.joblib'


# --- ЛОГИКА ---
def main():
    print(f"📂 Загружаем модель: {MODEL_PATH}")
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("❌ Модель не найдена.")
        return

    # 1. Загрузка данных
    print("📂 Читаем данные для валидации/калибровки...")
    try:
        df = pd.read_csv(DATA_FILE, decimal=',', sep=';', header=None, skiprows=1,
                         usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
        df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    except ValueError:
        # На случай если формат CSV чуть отличается
        print("⚠️ Ошибка чтения колонок, пробуем читать все...")
        df = pd.read_csv(DATA_FILE, decimal=',', sep=';')

    df['target'] = pd.to_numeric(df['target'], errors='coerce')
    df.dropna(subset=['target'], inplace=True)

    X, y = df.drop(columns=['target']), df['target']

    # Делим данные
    _, X_opt, _, y_opt = train_test_split(X, y, test_size=0.15, random_state=777)
    X_calib_new, X_test_final, y_calib_new, y_test_final = train_test_split(X_opt, y_opt, test_size=0.3,
                                                                            random_state=42)

    print(f"📊 Данные для новой калибровки: {len(X_calib_new)} строк")

    # --- ПУНКТ 3: ХИРУРГИЯ ВЕСОВ ---
    print("\n--- 🔧 ИНСПЕКЦИЯ ВЕСОВ (Stacking Ridge) ---")
    try:
        weights = model.final_estimator.coef_
        bias = model.final_estimator.intercept_
        model_names = [name for name, _, _ in model.configs] + ['MLP_Stack']

        print("Текущие веса моделей:")
        for name, w in zip(model_names, weights):
            print(f"  {name:10s}: {w:.4f}")

        has_negative = any(w < -0.2 for w in weights)
        if has_negative:
            print("\n⚠️ ОБНАРУЖЕНЫ ОТРИЦАТЕЛЬНЫЕ ВЕСА! Это может указывать на мультиколлинеарность.")
            print("🔄 ЗАМЕНЯЕМ Ridge на AverageBlender (простое среднее)...")
            model.final_estimator = AverageBlender()
            print("✅ Готово. Теперь модель использует среднее арифметическое предсказаний.")
        else:
            print("\n✅ Веса выглядят адекватно. Оставляем Ridge.")
    except AttributeError:
        print("⚠️ Не удалось извлечь веса (возможно, уже не Ridge). Пропускаем.")

    # --- ПУНКТ 2: МАССШТАБНАЯ ПЕРЕКАЛИБРОВКА ---
    print("\n--- 🎯 ПЕРЕКАЛИБРОВКА (Isotonic) ---")
    model.calibrate(X_calib_new, y_calib_new)
    print("✅ Калибровка завершена.")

    # --- ПРОВЕРКА ---
    print("\n--- 🏁 ФИНАЛЬНЫЙ ТЕСТ ---")
    y_pred = model.predict(X_test_final)
    rmse = np.sqrt(mean_squared_error(y_test_final, y_pred))
    r2 = r2_score(y_test_final, y_pred)
    print(f"📊 NEW RMSE: {rmse:.5f}")
    print(f"📊 NEW R2:   {r2:.5f}")

    # Сохраняем (ИСПРАВЛЕНО ТУТ)
    save_path = NEW_MODEL_NAME  # Сохраняем в текущую папку
    joblib.dump(model, save_path)
    print(f"\n💾 Оптимизированная модель сохранена: {os.path.abspath(save_path)}")


if __name__ == "__main__":
    main()