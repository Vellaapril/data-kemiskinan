# linear_regression.py
# Deskripsi: Prediksi nilai kemiskinan (Poverty) menggunakan Linear Regression.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load Data
df = pd.read_csv("C:/Users/vella/OneDrive/Dokumen/Kuliah/04TPLE009/Data Mining/Python/data jurnal/data_kemiskinan.csv")
df.columns = df.columns.str.strip()

# 2. Persiapan Data
X = df[['Growth', 'Unemployment']]
y = df['Poverty']

# 3. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 4. Latih Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Prediksi dan Evaluasi
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# 6. Visualisasi Prediksi vs Aktual
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred, color='darkorange', edgecolors='black')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel("Aktual Poverty")
plt.ylabel("Prediksi Poverty")
plt.title("Prediksi vs Aktual - Linear Regression")
plt.tight_layout()
plt.savefig("linear_regression_scatter.png")
plt.show()
