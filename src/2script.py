# Этот скрипт предполагает, что df из Шага 1 уже загружен
# Если нет, раскомментируйте первые 4 строки из Шага 1

print("\n" + "="*50)
print("### 3. Модель с логарифмическими параметрами ###")
print("="*50)

# Создаем копию df, чтобы не испортить оригинал
df_log = df.copy()

# Логарифмируем параметры с большими значениями
# Мы добавляем 1 (np.log1p), чтобы избежать ошибки, если вдруг где-то есть 0
df_log['log_col_4'] = np.log1p(df_log['col_4'])
df_log['log_col_5'] = np.log1p(df_log['col_5'])

# Наши новые фичи.
# col_2 и col_3 оставляем как есть, т.к. их масштаб небольшой
new_features = ['col_2', 'col_3', 'log_col_4', 'log_col_5']
if 'col_1' in features:
    new_features.insert(0, 'col_1')

y_log = df_log['col_6']
X_log = df_log[new_features]
X_log_with_const = sm.add_constant(X_log)

# Строим новую, "нелинейную" модель
log_model = sm.OLS(y_log, X_log_with_const).fit()

print(log_model.summary())