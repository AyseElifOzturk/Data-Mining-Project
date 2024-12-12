import pandas as pd

file_path = 'MushroomDataset/secondary_data.csv'
df = pd.read_csv(file_path, sep=';')

# Eksik verileri doldurma
df['cap-surface'].fillna(df['cap-surface'].mode()[0], inplace=True)
df['stem-width'].fillna(df['stem-width'].mean(), inplace=True)

# Çok eksik veri içeren sütunları kaldırma
df.drop(columns=['veil-type', 'stem-root'], inplace=True)

# Temizlenmiş veri setini kontrol etme
print(df.info())
