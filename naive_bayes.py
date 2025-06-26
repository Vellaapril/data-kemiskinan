# naive_bayes.py
# Deskripsi: Menerapkan model Naive Bayes (GaussianNB) untuk klasifikasi dan menampilkan visualisasi.

# 1. Import Pustaka
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 2. Memuat Data dari CSV
df = pd.read_csv("C:/Users/vella/OneDrive/Dokumen/Kuliah/04TPLE009/Data Mining/Python/data jurnal/data_kemiskinan.csv")
df.columns = df.columns.str.strip()  
print("Kolom yang tersedia dalam data:")
print(df.columns)

# 3. Persiapan Variabel dan Transformasi Target
X = df[['Growth', 'Unemployment']]
bins = [0, 15, 20, float('inf')]
labels = ['Rendah', 'Sedang', 'Tinggi']
y_categorical = pd.cut(df['Poverty'], bins=bins, labels=labels, right=False)

print("\n--- Transformasi Target (Poverty) ---")
print("Distribusi kategori kemiskinan yang baru dibuat:")
print(y_categorical.value_counts())
print("-" * 35)

# 🎨 Gambar 1: Visualisasi Distribusi Kategori Kemiskinan
plt.figure(figsize=(6, 4))
y_categorical.value_counts().plot(kind='bar', color='coral', edgecolor='black')
plt.title("Distribusi Kategori Kemiskinan")
plt.xlabel("Kategori")
plt.ylabel("Jumlah Tahun")
plt.tight_layout()
plt.savefig("distribusi_kemiskinan.png")  # simpan gambar
plt.show()

# 4. Pembagian Data (Train & Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.3, random_state=42, stratify=y_categorical
)

# 5. Inisialisasi dan Pelatihan Model
model = GaussianNB()
model.fit(X_train, y_train)

# 6. Evaluasi Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred, labels=labels)

print("\n--- Hasil Klasifikasi Naive Bayes ---")
print(f"Akurasi Model: {accuracy:.2f}")
print("\nLaporan Klasifikasi:")
print(class_report)
print("\nConfusion Matrix:")
print(pd.DataFrame(conf_matrix, index=labels, columns=labels))

# 🎨 Gambar 2: Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix Naive Bayes")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")  # simpan gambar
plt.show()

# 7. Contoh Prediksi
data_baru = pd.DataFrame({'Growth': [5.0], 'Unemployment': [7.0]})
prediksi_kategori = model.predict(data_baru)
prediksi_probabilitas = model.predict_proba(data_baru)

print("\n--- Contoh Prediksi ---")
print(f"Data input: Pertumbuhan = 5.0%, Pengangguran = 7.0%")
print(f"Prediksi Kategori Kemiskinan: '{prediksi_kategori[0]}'")
print("Probabilitas untuk setiap kategori:")
for label, proba in zip(model.classes_, prediksi_probabilitas[0]):
    print(f"  - {label}: {proba:.2%}")
