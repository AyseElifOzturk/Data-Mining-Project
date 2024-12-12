import pandas as pd
file_path = 'MushroomDataset/secondary_data.csv'
# Dosyanın ayırıcı karakterini belirtiyoruz
df = pd.read_csv(file_path, sep=';')
print(df.info())
# Eksik veri içeren satırları kaldırıyoruz
df_cleaned = df.dropna()
# Temizlenmiş DataFrame'i görüntülüyoruz
print(df_cleaned)
