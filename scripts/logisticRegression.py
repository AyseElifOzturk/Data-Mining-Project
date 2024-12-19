# Gerekli kütüphaneleri import etme
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Temizlenmiş veri setini yükleme
file_path = 'MushroomDataset/cleanedMushroom.csv'
df = pd.read_csv(file_path)

# Kategorik verileri sayısallaştırma
df_encoded = pd.get_dummies(df, drop_first=True)  # Kategorik sütunları dönüştür ve bir sütunu referans olarak al

# Bağımlı (target) ve bağımsız (features) değişkenleri belirleme
X = df_encoded.drop(columns=['class_p'])  # 'class_p' hedef sütunun sayısallaştırılmış hali
y = df_encoded['class_p']                # Hedef değişken

# Veri setini eğitim ve test olarak ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression modelini oluşturma ve eğitme
model = LogisticRegression(max_iter=1000)  # Iterasyon sayısını artırarak convergence hatalarını önleyin
model.fit(X_train, y_train)

# Test veri seti üzerinde tahmin yapma
y_pred = model.predict(X_test)

# Modelin başarımını değerlendirme
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Sonuçları yazdırma
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", class_report)
