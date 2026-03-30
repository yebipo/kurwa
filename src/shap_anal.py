    # ===================================================================================
    # ===   СКРИПТ ДЛЯ АНАЛИЗА ГОТОВОЙ "ЧЕМПИОНСКОЙ" МОДЕЛИ С ПОМОЩЬЮ SHAP      ===
    # ===                            (V2.5 - ИСПРАВЛЕНА ОШИБКА ЧТЕНИЯ CSV)            ===
    # ===================================================================================

    import pandas as pd
    import joblib
    import shap
    from sklearn.model_selection import train_test_split
    import warnings
    import time

    # Импорты нужны, чтобы pickle мог восстановить классы
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LinearRegression
    import lightgbm as lgb
    import xgboost as xgb
    import catboost as cb

    warnings.filterwarnings('ignore')


    # Класс-заглушка для успешной загрузки объекта FinalModel
    class FinalModel:
        pass


    # --- 1. КОНФИГУРАЦИЯ ---
    MODEL_PATH = 'model.joblib'  # Ваша модель (скорее всего, the_best_model_ever.joblib)
    FULL_DATA_PATH = 'data_with_clusters.csv'  # Файл с 5-ю колонками, созданный на прошлом шаге

    ANALYSIS_DATA_RATIO = 0.3
    RANDOM_STATE = 42

    print("--- Запуск скрипта для интерпретации модели ---")
    start_time = time.time()

    # --- 2. ЗАГРУЗКА МОДЕЛИ И ДАННЫХ ---
    try:
        print(f"\n[Шаг 1/5] Загрузка модели-контейнера из '{MODEL_PATH}'...")
        model_container = joblib.load(MODEL_PATH)
        actual_model = model_container.stacking_model
        print("   ✅ Успешно извлечена реальная модель ('stacking_model') из контейнера!")

        print(f"\n[Шаг 2/5] Загрузка полного набора данных из '{FULL_DATA_PATH}'...")
        # =============================================================================
        # ===            ФИНАЛЬНОЕ ИСПРАВЛЕНИЕ: Читаем CSV с разделителем-запятой    ===
        # =============================================================================
        full_dataset = pd.read_csv(FULL_DATA_PATH)

        print(f"   ✅ Данные загружены. Всего строк: {len(full_dataset)}. Признаки: {full_dataset.columns.tolist()}")

        # Проверка на правильное количество колонок
        if len(full_dataset.columns) != 5:
            print(f"\n❌ ОШИБКА: Ожидалось 5 колонок в файле, но найдено {len(full_dataset.columns)}.")
            print("   Пожалуйста, пересоздайте файл 'data_with_clusters.csv' с помощью скрипта 'generate_features.py'.")
            exit()

    except Exception as e:
        import traceback

        print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА: {e}")
        traceback.print_exc()
        exit()

    # --- 3. РАЗДЕЛЕНИЕ ДАННЫХ ДЛЯ ЦЕЛЕЙ АНАЛИЗА ---
    print(f"\n[Шаг 3/5] Разделение данных для SHAP...")
    X_background_full, X_for_analysis = train_test_split(
        full_dataset, test_size=ANALYSIS_DATA_RATIO, random_state=RANDOM_STATE
    )
    print(f"   ✅ Данные разделены.")

    # --- 4. РАСЧЕТ SHAP-ЗНАЧЕНИЙ ---
    print("\n" + "=" * 60)
    print("###          АНАЛИЗ МОДЕЛИ С ПОМОЩЬЮ SHAP           ###")
    print("=" * 60)
    X_background_sample = shap.sample(X_background_full, 100, random_state=RANDOM_STATE)
    print(f"\n[Шаг 4/5] Создание объяснителя и расчет SHAP-значений...")
    print("   (Это самый долгий этап, он может занять несколько минут. Пожалуйста, подождите...)")

    explainer = shap.KernelExplainer(actual_model.predict, X_background_sample)
    shap_values = explainer.shap_values(X_for_analysis)
    print("\n   ✅ Расчет SHAP-значений завершен!")

    # --- 5. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (без изменений) ---
    print(f"\n[Шаг 5/5] Генерация и отображение графиков анализа...")
    print("\n--> Отображается [Анализ 1]: Глобальная важность...")
    shap.summary_plot(shap_values, X_for_analysis, plot_type="bar", show=True)
    print("    Закройте окно с графиком, чтобы продолжить...")
    print("\n--> Отображается [Анализ 2]: Карта влияния...")
    shap.summary_plot(shap_values, X_for_analysis, show=True)
    print("    Закройте окно с графиком, чтобы продолжить...")
    i = 0
    print(f"\n--> Отображается [Анализ 3]: Разбор прогноза для объекта {i}...")
    shap.force_plot(explainer.expected_value, shap_values[i, :], X_for_analysis.iloc[i, :], matplotlib=True, show=True)
    print("    Закройте окно с графиком, чтобы продолжить...")
    interesting_feature = X_for_analysis.columns[-1]
    print(f"\n--> Отображается [Анализ 4]: Влияние признака '{interesting_feature}'...")
    shap.dependence_plot(interesting_feature, shap_values, X_for_analysis, interaction_index=None, show=True)
    print("    Закройте окно с графиком для завершения.")

    end_time = time.time()
    print(f"\n--- Анализ полностью завершен за {end_time - start_time:.2f} секунд ---")