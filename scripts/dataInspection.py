import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
print("Tüm kütüphaneler başarıyla yüklendi!")

# Veri setini yükleme
file_path = 'MushroomDataset/secondary_data.csv'
df = pd.read_csv(file_path)

# Veri setinin ilk birkaç satırını görüntüleyin
print(df.head())
# Eksik veri kontrolü
print(df.isnull().sum())

# Eksik yok
