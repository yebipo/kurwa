import pandas as pd

file_adequate = 'output_adequate.csv'
file_million = 'million.csv'
file_output = 'million_unique_only.csv'

print("Загрузка...")
df_adequate = pd.read_csv(file_adequate, sep=';', dtype=str)
df_million = pd.read_csv(file_million, sep=';', dtype=str)

print("Ищем строки из миллионника, которых нет в адекватном...")
# Теперь million — основной (левый), а adequate — проверочный (правый)
df_diff = df_million.merge(df_adequate, how='left', indicator=True)
result = df_diff[df_diff['_merge'] == 'left_only'].drop('_merge', axis=1)

print(f"Сохранение... Уникальных строк в миллионнике: {len(result)}")
result.to_csv(file_output, index=False, sep=';', encoding='utf-8-sig')