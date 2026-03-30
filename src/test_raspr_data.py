# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# --- КОНФИГУРАЦИЯ ---
DATA_FILE = 'million.csv'  # Файл с данными

def load_data():
    """Загрузка данных из файла million.csv."""
    if not os.path.exists(DATA_FILE):
        print(f"❌ Ошибка: Файл {DATA_FILE} не найден! Положите его рядом со скриптом.")
        return None

    # Используем ту же логику загрузки, что и в основном скрипте
    try:
        df = pd.read_csv(DATA_FILE, decimal=',', sep=';', header=None, skiprows=1,
                         usecols={1: 'f1', 2: 'f2', 3: 'f3', 4: 'f4', 5: 'target'}.keys())
        df.columns = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'target']
        df['target'] = pd.to_numeric(df['target'], errors='coerce')
        df.dropna(subset=['target'], inplace=True)
        print(f"✅ Данные успешно загружены. Размер: {df.shape}")
        return df
    except Exception as e:
        print(f"❌ Ошибка при загрузке или парсинге данных: {e}")
        return None

def analyze_data(df):
    """Выполняет запрошенный анализ данных."""

    print("\n" + "="*50)
    print("        1. Общая статистика признаков (X)")
    print("="*50)

    # 1. Общая статистика признаков (X)
    X_stats = df[['feature_1', 'feature_2', 'feature_3', 'feature_4']].describe().T
    print(X_stats)

    print("\n" + "="*50)
    print("        2. Анализ целевой переменной (target)")
    print("="*50)

    # 2a. Процент крайних значений target
    n_rows = len(df)
    low_cutoff = 0.01
    high_cutoff = 0.99
    n_low = (df['target'] < low_cutoff).sum()
    n_high = (df['target'] > high_cutoff).sum()

    print(f"Минимальное значение target: {df['target'].min():.4f}")
    print(f"Максимальное значение target: {df['target'].max():.4f}")
    print(f"Среднее значение target: {df['target'].mean():.4f}")
    print(f"Медиана target: {df['target'].median():.4f}")
    print(f"---")
    print(f"Количество значений < {low_cutoff:.2f} (близких к 0): {n_low} ({n_low / n_rows * 100:.2f}%)")
    print(f"Количество значений > {high_cutoff:.2f} (близких к 1): {n_high} ({n_high / n_rows * 100:.2f}%)")

    # 2b. Гистограмма target (Визуализация)
    plt.figure(figsize=(10, 6))
    sns.histplot(df['target'], bins=50, kde=True)
    plt.title('Распределение целевой переменной (Target)')
    plt.xlabel('Target Value')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.show()


    print("\n" + "="*50)
    print("        3. Матрица корреляции")
    print("="*50)

    # 3. Матрица корреляции
    corr_matrix = df.corr()
    print(corr_matrix)

    # Визуализация корреляции
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Матрица корреляции признаков и Target')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    df = load_data()
    if df is not None:
        analyze_data(df)