import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# --- НАСТРОЙКИ ---
INPUT_FILE = 'unique_rows.csv'
# Берем твою последнюю плавную модель
MODEL_FILE = "final_model_smooth.joblib"

CSV_SEP = ';'
CSV_DECIMAL = ','


# --- КЛАССЫ (НУЖНЫ ДЛЯ ЗАГРУЗКИ) ---
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
        self.model = None

    def predict(self, X_raw):
        X_ready = self.preprocessor.transform(X_raw)
        return self.model.predict(X_ready)


class PlattCalibrator:
    def __init__(self):
        self.a_ = None
        self.b_ = None

    def predict(self, X):
        linear = self.a_ * X.flatten() + self.b_
        return np.clip(1.0 / (1.0 + np.exp(-linear)), 0, 1).reshape(X.shape)


class CalibratedProductionModel:
    def __init__(self, base_model, calibrator):
        self.base_model = base_model
        self.calibrator = calibrator

    def predict(self, X_raw):
        raw_pred = self.base_model.predict(X_raw)
        return self.calibrator.predict(raw_pred)


# --- АНАЛИЗ ---

def main():
    print("🔍 Анализ гетероскедастичности...")

    # 1. Загрузка данных
    df = pd.read_csv(INPUT_FILE, sep=CSV_SEP, decimal=CSV_DECIMAL, header=None, on_bad_lines='skip', engine='python')
    df = df[[1, 2, 3, 4, 5]]
    df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    # 2. Загрузка модели и предсказание
    model = joblib.load(MODEL_FILE)
    X = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']]
    y_true = df['target'].values

    print("🧠 Считаем предсказания...")
    y_pred = model.predict(X)

    # Считаем остатки (ошибки)
    residuals = y_pred - y_true
    # Считаем модуль ошибки (индикатор неопределенности)
    abs_residuals = np.abs(residuals)

    # 3. Визуализация
    features = ['feature_1', 'feature_2', 'feature_3', 'feature_4']
    plt.figure(figsize=(20, 10))

    for i, col in enumerate(features):
        plt.subplot(2, 2, i + 1)
        # Берем сэмпл для скорости, если данных слишком много
        sample_idx = np.random.choice(len(df), min(10000, len(df)), replace=False)

        # Строим график: значение признака vs Ошибка
        sns.regplot(x=X.iloc[sample_idx][col], y=residuals[sample_idx],
                    scatter_kws={'alpha': 0.1, 's': 2},
                    line_kws={'color': 'red'},
                    lowess=True)  # Lowess покажет нелинейный тренд ошибки

        plt.axhline(0, color='black', linestyle='--', alpha=0.5)
        plt.title(f'Residuals vs {col}')
        plt.ylabel('Error (Pred - Real)')

    plt.tight_layout()
    plt.show()

    # 4. Математический чекап: корреляция признаков с абсолютной ошибкой
    print("\n📊 Корреляция признаков с АБСОЛЮТНОЙ ошибкой (неуверенностью):")
    for col in features:
        corr = np.corrcoef(X[col], abs_residuals)[0, 1]
        print(f"   {col:<10} : {corr:+.4f}")


if __name__ == "__main__":
    main()