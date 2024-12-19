import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Temizlenmiş veri setini yükleme
file_path = 'MushroomDataset/cleanedMushroom.csv'
df = pd.read_csv(file_path)

# 2. Bağımlı ve bağımsız değişkenlerin ayrılması
X = df.drop(columns=['class'])  # 'class' sınıflandırmak istediğimiz sütun
y = df['class']

# 3. Kategorik değişkenleri sayısal hale getirme
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':  # Kategorik sütunlar
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le  # İleride ters çevirme için kaydetme

# Sınıf sütununu da encode etme
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# 4. Veriyi eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 5. SVM sınıflandırıcısını oluşturma ve eğitme
svm_model = SVC(kernel='linear', random_state=42)  # Linear kernel kullandık
svm_model.fit(X_train, y_train)

# 6. Modeli test etme
y_pred = svm_model.predict(X_test)

# 7. Doğruluk ve metrikleri yazdırma
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Modeli Doğruluğu: {accuracy:.2f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=le_y.classes_))

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_y.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# 9. Çapraz doğrulama
cv_scores = cross_val_score(svm_model, X, y, cv=10)  # 10 katlı çapraz doğrulama
print(f"Çapraz Doğrulama Doğruluk Skorları: {cv_scores}")
print(f"Ortalama Doğruluk: {cv_scores.mean():.2f}")
