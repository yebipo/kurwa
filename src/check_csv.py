import pandas as pd

# Грузим данные
train_df = pd.read_csv('million.csv', sep=';', decimal=',', on_bad_lines='skip')
test_df = pd.read_csv('unique_rows.csv', sep=';', decimal=',', on_bad_lines='skip', header=None)

print("=== КАК ВЫГЛЯДИТ TRAIN (million.csv) ===")
print(train_df.head(3))
print(f"Всего колонок: {train_df.shape[1]}")

print("\n=== КАК ВЫГЛЯДИТ TEST (unique_rows.csv) ===")
print(test_df.head(3))
print(f"Всего колонок: {test_df.shape[1]}")