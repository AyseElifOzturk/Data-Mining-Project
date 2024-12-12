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

# Temizlenmiş veri setinin genel bilgilerini görüntüleme
print("\nTemizlenmiş veri seti bilgileri:")
print(df_cleaned.info())
