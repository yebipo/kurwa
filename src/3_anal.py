import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings('ignore')

# --- НАСТРОЙКИ ---
FILE_PATH = "3func.csv"

# Какие колонки анализировать (по номеру, начиная с 0)
# Если в Excel это колонки 2, 3, 4, 5, то здесь индексы: 1, 2, 3, 4
TARGET_INDICES = [1, 2, 3, 4]


def analyze_weights(series, col_name):
    print(f"\n{'=' * 60}")
    print(f"⚖️ АНАЛИЗ ВЕСОВ: {col_name}")

    # Удаляем пустые
    series = series.dropna()

    if len(series) == 0:
        print("   (Пустая колонка)")
        return

    # Проверяем уникальные
    counts = series.value_counts(normalize=True).sort_values(ascending=False)
    unique_count = len(counts)

    # --- ЕСЛИ ЭТО КОНСТАНТА ---
    if unique_count == 1:
        val = counts.index[0]
        print(f"📌 КОНСТАНТА. Всегда равно: {int(val)}")
        return

    # --- ЕСЛИ ЭТО КАТЕГОРИЯ (Feature 3, степени двойки и т.д.) ---
    # Если уникальных значений мало (меньше 100), мы можем дать веса
    if unique_count < 100:
        print(f"📊 КАТЕГОРИЯ (Всего {unique_count} вариантов). Готовые списки:")

        # Формируем списки для копирования в Python
        values_list = counts.index.astype(int).tolist()
        probs_list = counts.values.round(4).tolist()  # Округляем вероятности

        # Нормировка суммы вероятностей ровно до 1.0 (на всякий случай)
        probs_list[-1] = round(1.0 - sum(probs_list[:-1]), 4)

        print(f"\n# Скопируйте это в генератор:")
        print(f"{col_name}_VALUES = {values_list}")
        print(f"{col_name}_PROBS  = {probs_list}")

        print(f"\n   Топ-5 лидеров:")
        print(counts.head(5).to_string())
        return

    # --- ЕСЛИ ЭТО НЕПРЕРЫВНОЕ (Feature 4) ---
    print(f"📈 НЕПРЕРЫВНОЕ. (Много вариантов: {unique_count})")
    print(f"   Мин: {series.min():_.0f}")
    print(f"   Макс: {series.max():_.0f}")
    print(f"   Среднее: {series.mean():_.0f}")
    print(f"   Медиана: {series.median():_.0f}")

    # Разбиваем на диапазоны (бины), чтобы понять, где густо
    try:
        bins = pd.cut(series, bins=5).value_counts(normalize=True).sort_index()
        print("\n   Распределение по диапазонам:")
        print(bins)
    except:
        pass


def main():
    print("=== ИЗВЛЕЧЕНИЕ ВЕСОВ ИЗ КОЛОНОК [2-5] ===")

    if not os.path.exists(FILE_PATH):
        print(f"❌ Файл {FILE_PATH} не найден.")
        return

    # Читаем с error_bad_lines=False (или on_bad_lines='skip'), чтобы пропустить битые строки
    # usecols=None, читаем всё, потом обрежем
    try:
        # Читаем как строки, чтобы не упало на парсинге, потом конвертируем
        df = pd.read_csv(FILE_PATH, sep=';', decimal=',', dtype=str, on_bad_lines='skip', engine='python')
    except Exception as e:
        print(f"❌ Ошибка чтения: {e}")
        return

    print(f"📂 Загружено строк: {len(df)}")

    # Переименовываем для удобства
    df.columns = [f"Col_{i + 1}" for i in range(len(df.columns))]

    # Проходим по нужным индексам
    for idx in TARGET_INDICES:
        if idx < len(df.columns):
            col_name = df.columns[idx]

            # Конвертируем в числа, ошибки -> NaN
            series_numeric = pd.to_numeric(df[col_name], errors='coerce')

            analyze_weights(series_numeric, col_name)
        else:
            print(f"\n⚠️ Индекс {idx} выходит за пределы файла.")


if __name__ == "__main__":
    main()