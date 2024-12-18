import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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

# Random Forest sınıflandırıcısını oluşturma
rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf_clf.fit(X_train, y_train)

# Modeli test etme
y_pred = rf_clf.predict(X_test)

# Doğruluk ve metrikleri yazdırma
accuracy = accuracy_score(y_test, y_pred)
print(f"Random Forest Modeli Doğruluğu: {accuracy:.2f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=le_y.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le_y.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Çapraz doğrulama
cv_scores = cross_val_score(rf_clf, X, y, cv=10)  # 10 katlı çapraz doğrulama
print(f"Çapraz Doğrulama Doğruluk Skorları: {cv_scores}")
print(f"Ortalama Doğruluk: {cv_scores.mean():.2f}")

# !!!!!!!
# Özellik önem derecelerini hesaplama
feature_importances = rf_clf.feature_importances_
features = X.columns

# Özellik önemlerini sıralama
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Görselleştirme
plt.figure(figsize=(12, 8))
plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10], color='skyblue')
plt.gca().invert_yaxis()
plt.title("10 most important feature")
plt.xlabel("Importance level")
plt.ylabel("Features")
plt.show()

# İlk 10 önemli özelliği seçme
top_features = importance_df['Feature'][:10].tolist()

# Yeni veri kümesi oluşturma (en önemli 10 özellik)
X_selected = X[top_features]

# Modeli tekrar eğitmek için hazırlanma
X_train_sel, X_test_sel, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)

# Yeni Random Forest modeli oluşturma
rf_clf_selected = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
rf_clf_selected.fit(X_train_sel, y_train)

# Test doğruluğu ve çapraz doğrulama skorlarını hesaplama
y_pred_sel = rf_clf_selected.predict(X_test_sel)
accuracy_selected = accuracy_score(y_test, y_pred_sel)
cv_scores_selected = cross_val_score(rf_clf_selected, X_selected, y, cv=10)

#accuracy_selected, cv_scores_selected.mean()

# Sonuçları yazdırma
print("\n--- Sonuçlar (Özellik Seçimi Sonrası) ---")
print(f"Test Setindeki Doğruluk: {accuracy_selected:.2f}")
print(f"Çapraz Doğrulama Ortalama Doğruluk: {cv_scores_selected.mean():.2f}")
print("\nÇapraz Doğrulama Tüm Skorları:")
print(cv_scores_selected)

