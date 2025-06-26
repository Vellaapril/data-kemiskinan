# decision_tree.py
# Deskripsi: Klasifikasi kategori kemiskinan menggunakan Decision Tree.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load Data
df = pd.read_csv("C:/Users/vella/OneDrive/Dokumen/Kuliah/04TPLE009/Data Mining/Python/data jurnal/data_kemiskinan.csv")
df.columns = df.columns.str.strip()

# 2. Persiapan Data
X = df[['Growth', 'Unemployment']]
bins = [0, 15, 20, float('inf')]
labels = ['Rendah', 'Sedang', 'Tinggi']
y = pd.cut(df['Poverty'], bins=bins, labels=labels, right=False)

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 4. Inisialisasi dan Latih Model
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluasi
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi Decision Tree: {acc:.2f}")
print("\nLaporan Klasifikasi:")
print(classification_report(y_test, y_pred, zero_division=0))

# 6. Confusion Matrix
conf = confusion_matrix(y_test, y_pred, labels=labels)
sns.heatmap(conf, annot=True, fmt='d', cmap='Greens',
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Decision Tree")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.savefig("decision_tree_confusion_matrix.png")
plt.show()

# 7. Visualisasi Pohon Keputusan
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=['Growth', 'Unemployment'], class_names=labels, filled=True)
plt.title("Visualisasi Decision Tree")
plt.savefig("decision_tree_structure.png")
plt.show()
