import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Contoh data untuk analisis
# Buat data yang mirip dengan gambar yang diberikan
usia = np.array([23, 25, 26, 27, 29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40])
gaji = np.array([3000, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 4000, 4100, 4200, 4300, 4400, 4500, 4600])
jenis_kelamin = np.array(['pria', 'wanita', 'pria', 'wanita', 'pria', 'wanita', 'pria', 'wanita', 'pria', 'wanita', 'pria', 'wanita', 'pria', 'wanita', 'pria'])

data = pd.DataFrame({'Usia': usia, 'Gaji': gaji, 'Jenis Kelamin': jenis_kelamin})

# Scatter plot Usia vs Gaji
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.scatter(data['Usia'], data['Gaji'], color='blue', alpha=0.6)
plt.title('Scatter Plot')
plt.xlabel('Usia')
plt.ylabel('Gaji')

# Histogram Gaji
plt.subplot(2, 2, 2)
plt.hist(data['Gaji'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Gaji')
plt.xlabel('Gaji')
plt.ylabel('Frequency')

# Box plot Usia
plt.subplot(2, 2, 3)
sns.boxplot(y=data['Usia'])
plt.title('Box Plot of Usia')
plt.ylabel('Usia')

# Bar plot Jenis Kelamin
plt.subplot(2, 2, 4)
sns.countplot(data['Jenis Kelamin'], palette=['blue', 'pink'])
plt.title('Barplot Jenis Kelamin')
plt.xlabel('Jenis Kelamin')
plt.ylabel('Jumlah')

plt.tight_layout()
plt.show()

# Pertanyaan 1: Regresi Analisis
# 1.1. Membuat model regresi linier
model = LinearRegression()
X = data['Usia'].values.reshape(-1, 1)
y = data['Gaji']
model.fit(X, y)

# 1.2. Menampilkan parameter regresi
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

# 1.3. Evaluasi model
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Pertanyaan 2: Korelasi
# 2.1. Menghitung korelasi
correlation, p_value = pearsonr(data['Usia'], data['Gaji'])
print("Pearson Correlation:", correlation)
print("P-value:", p_value)

# Pertanyaan 3: Interpretasi dan Kesimpulan
# 3.1. Histogram Gaji
plt.hist(data['Gaji'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
plt.title('Histogram of Gaji')
plt.xlabel('Gaji')
plt.ylabel('Frequency')
plt.show()

# 3.2. Box plot Usia
sns.boxplot(y=data['Usia'])
plt.title('Box Plot of Usia')
plt.ylabel('Usia')
plt.show()

# 3.3. Bar plot Jenis Kelamin
sns.countplot(data['Jenis Kelamin'], palette=['blue', 'pink'])
plt.title('Barplot Jenis Kelamin')
plt.xlabel('Jenis Kelamin')
plt.ylabel('Jumlah')
plt.show()

# 3.4. Analisis terpisah berdasarkan Jenis Kelamin
for gender in data['Jenis Kelamin'].unique():
    subset = data[data['Jenis Kelamin'] == gender]
    correlation, p_value = pearsonr(subset['Usia'], subset['Gaji'])
    print(f"Correlation for {gender}: {correlation}, P-value: {p_value}")
