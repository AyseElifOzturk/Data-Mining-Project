import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Temizlenmiş veri setini yükleme
file_path = 'MushroomDataset/cleanedMushroom.csv'
df = pd.read_csv(file_path)

# Veri setinin genel bilgisi
info = df.info()
head = df.head()
summary = df.describe(include='all')

info, head, summary

# Bağımlı ve bağımsız değişkenlerin ayrılması
X = df.drop(columns=['class'])  # 'class' sınıflandırmak istediğimiz sütun
y = df['class']

# Kurallara dayalı sınıflandırma
def rule_based_classifier(row):
    # Eğer şapka çapı büyük (>10 cm) ve şapka şekli 'convex' ise zehirli
    if row['cap-diameter'] > 10 and row['cap-shape'] in ['x']:  # x: convex
        return 'poisonous'
    # Eğer şapka yüzeyi 'fibrous' ve şapka rengi 'red' ise zehirli
    elif row['cap-surface'] == 'f' and row['cap-color'] == 'r':
        return 'poisonous'
    # Eğer mantar kanar veya zedelenirse 'poisonous'
    elif row['does-bruise-or-bleed'] == 't':
        return 'poisonous'
    # Eğer lamel rengi beyaz veya kahverengi ve şapka rengi kahverengiyse yenilebilir
    elif row['gill-color'] in ['w', 'b'] and row['cap-color'] == 'b':
        return 'edible'
    # Eğer habitat 'meadow' ve mevsim yaz ise yenilebilir
    elif row['habitat'] == 'm' and row['season'] == 's':
        return 'edible'
    # Eğer mantar halkaya sahipse ve halka türü 'pendant' ise zehirli
    elif row['has-ring'] == 't' and row['ring-type'] == 'p':
        return 'poisonous'
    # Eğer sap yüksekliği > 15 cm ve sap rengi beyaz ise zehirli
    elif row['stem-height'] > 15 and row['stem-color'] == 'w':
        return 'poisonous'
    # Eğer habitat 'woods' ve şapka yüzeyi 'smooth' ise yenilebilir
    elif row['habitat'] == 'w' and row['cap-surface'] == 's':
        return 'edible'
    # Varsayılan olarak zehirli
    else:
        return 'poisonous'

# Sınıflandırıcıyı veri setine uygulama
y_pred = X.apply(rule_based_classifier, axis=1)

# Doğruluk ve metrikler
accuracy = accuracy_score(y, y_pred)
print(f"Rule-Based Model Doğruluğu: {accuracy:.2f}")
print("\nSınıflandırma Raporu:")
print(classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred, labels=y.unique())
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y.unique())
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
