import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.tree import plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Temizlenmiş veri setini yükleme
file_path = 'MushroomDataset/cleanedMushroom.csv'
df = pd.read_csv(file_path)

# Bağımlı ve bağımsız değişkenlerin ayrılması
X = df.drop(columns=['class'])  # 'class' sınıflandırmak istediğimiz sütun
y = df['class']

# Kategorik değişkenleri sayısal hale getirme
label_encoders = {}
for column in X.columns:
    if X[column].dtype == 'object':  # Kategorik sütunlar
        le = LabelEncoder()
        X[column] = le.fit_transform(X[column])
        label_encoders[column] = le  # İleride ters çevirme için kaydetme

# Sınıf sütununu da encode etme
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Veriyi eğitim ve test olarak bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Karar ağacı sınıflandırıcısını oluşturma (model karmaşıklığını azaltmak için max_depth kullanıldı)
clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(X_train, y_train)

# Modeli test etme
y_pred = clf.predict(X_test)

# Doğruluk ve metrikleri yazdırma
accuracy = accuracy_score(y_test, y_pred)
print(f"Karar Ağacı Modeli Doğruluğu: {accuracy:.2f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=le_y.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_y.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Çapraz doğrulama
tuned_cv_scores = cross_val_score(clf, X, y, cv=10)  # 10 katlı çapraz doğrulama
print(f"Çapraz Doğrulama Doğruluk Skorları: {tuned_cv_scores}")
print(f"Ortalama Doğruluk: {tuned_cv_scores.mean():.2f}")

# Karar ağacı görselleştirme
plt.figure(figsize=(15, 10))
plot_tree(clf, feature_names=X.columns, class_names=le_y.classes_, filled=True)
plt.show()
