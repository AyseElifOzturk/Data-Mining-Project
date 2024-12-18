import pandas as pd

# Doğru ayırıcıyı belirtmek için delimiter parametresini ekliyoruz
file_path = 'MushroomDataset/secondary_data.csv' 
df = pd.read_csv(file_path, delimiter=';')  # Ayırıcı olarak ';' kullanıyoruz

# Eksik veri sayısını analiz ediyoruz
missing_counts = df.isnull().sum()
print("Her sütundaki eksik veri sayısı:\n", missing_counts)

missing_percentages = (missing_counts / len(df)) * 100
print("\nHer sütundaki eksik veri yüzdesi:\n", missing_percentages)

# Çok eksik veri içeren sütunları belirleme (%50'den fazla eksik veri)
threshold = 50  # Eksik veri oranı eşiği
columns_to_drop = missing_percentages[missing_percentages > threshold].index
print("\n%50'den fazla eksik veri içeren sütunlar:\n", columns_to_drop)

# %50'den fazla eksik veri içeren sütunları kaldırma
df_cleaned = df.drop(columns=columns_to_drop)
print("\n%50'den fazla eksik veri içeren sütunlar kaldırıldı.")

# %50'den az eksik veri olan sütunları doldurma
for column in df_cleaned.columns:
    if df_cleaned[column].isnull().sum() > 0:  # Eksik veri içeren sütunlar
        if df_cleaned[column].dtype in ['float64', 'int64']:  # Sayısal sütunlar
            median_value = df_cleaned[column].median()
            df_cleaned[column] = df_cleaned[column].fillna(median_value)
            print(f"{column} sütunu medyan değeri ({median_value}) ile dolduruldu.")
        else:  # Kategorik sütunlar
            mode_value = df_cleaned[column].mode()[0]
            df_cleaned[column] = df_cleaned[column].fillna(mode_value)
            print(f"{column} sütunu mod değeri ({mode_value}) ile dolduruldu.")

# Temizlenmiş veri setinin genel bilgilerini görüntüleme
print("\nTemizlenmiş veri seti bilgileri:")
print(df_cleaned.info())

# Temizlenmiş veri setini kaydetme
output_path = 'MushroomDataset/cleanedMushroom.csv'
df_cleaned.to_csv(output_path, index=False)
print(f"\nTemizlenmiş veri seti '{output_path}' olarak kaydedildi.")
