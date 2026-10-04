
# ===== Data Preparation Lanjutan — Studi Kasus: Telco Customer Churn


## ===== 1. Import Library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE


## ===== 2. Load Dataset

# Di Google Colab: upload file WA_Fn-UseC_-Telco-Customer-Churn.csv terlebih dahulu,
# lalu sesuaikan path di bawah ini.
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
print("Bentuk data:", df.shape)
df.head()


## ===== 3. Diagnosis Masalah Data

print("Duplikat:", df.duplicated().sum())

# TotalCharges terbaca sebagai teks karena ada sel kosong berupa spasi ' '
print("\nContoh nilai aneh di TotalCharges:",
      df.loc[pd.to_numeric(df['TotalCharges'], errors='coerce').isna(), 'TotalCharges'].unique())

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("\nMissing values per kolom:")
print(df.isnull().sum()[df.isnull().sum() > 0])

print("\nDistribusi target (Churn):")
print(df['Churn'].value_counts())
print(df['Churn'].value_counts(normalize=True).round(3))


**Hasil diagnosis:**


## ===== 4. Preprocessing (Aturan Aman)

# 1. Bersihkan & siapkan target
df = df.drop(columns=['customerID'])
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# 2. One-hot encoding
X = pd.get_dummies(df.drop(columns=['Churn']), drop_first=True)
y = df['Churn']
print("Jumlah fitur setelah encoding:", X.shape[1])

# 3. Split LEBIH DULU (stratified agar proporsi churn terjaga)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
print("Train:", X_train.shape, dict(y_train.value_counts()))
print("Test :", X_test.shape, dict(y_test.value_counts()))

# 4. KNN Imputer — hanya belajar dari train
num_cols = X_train.select_dtypes(['int64', 'float64']).columns.tolist()
imputer = KNNImputer(n_neighbors=5)
X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
X_test[num_cols] = imputer.transform(X_test[num_cols])
print("\nMissing setelah imputasi:", X_train.isnull().sum().sum(), "(train),",
      X_test.isnull().sum().sum(), "(test)")

# 5. Scaling — hanya belajar dari train
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 6. Deteksi outlier di train
iso = IsolationForest(contamination=0.02, random_state=42)
is_outlier = iso.fit_predict(X_train)
n_outlier = (is_outlier == -1).sum()
print(f"\nOutlier terdeteksi: {n_outlier} baris ({n_outlier/len(X_train)*100:.1f}%)")
print("Rata-rata tenure outlier :", round(X_train.loc[is_outlier == -1, 'tenure'].mean(), 1))
print("(Outlier di sini = pelanggan tenure sangat panjang — kejadian sah, jadi DIPERTAHANKAN, bukan dihapus.)")


## ===== 5. Training Model (Baseline vs SMOTE)

# Baseline
model_base = LogisticRegression(max_iter=2000).fit(X_train, y_train)

# SMOTE hanya pada train
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("Distribusi train sesudah SMOTE:", dict(pd.Series(y_train_sm).value_counts()))

model_smote = LogisticRegression(max_iter=2000).fit(X_train_sm, y_train_sm)


## ===== 6. Evaluasi Model (pada test ASLI yang tidak seimbang)

for nama, model in [("Baseline (tanpa SMOTE)", model_base),
                       ("Dengan SMOTE", model_smote)]:
    y_pred = model.predict(X_test)
    print(f"--- {nama} ---")
    print("Akurasi :", round(accuracy_score(y_test, y_pred), 3))
    print("F1-Score:", round(f1_score(y_test, y_pred), 3))
    print(classification_report(y_test, y_pred, target_names=['Tidak Churn', 'Churn']))


## ===== 7. Visualisasi

# Distribusi kelas sebelum vs sesudah SMOTE
before = y_train.value_counts().sort_index()
after = pd.Series(y_train_sm).value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(9, 4))
axes[0].bar(['Tidak Churn', 'Churn'], before.values, color=['#2E5395', '#C0504D'])
axes[0].set_title('Sebelum SMOTE (data latih)')
axes[0].set_ylabel('Jumlah sampel')
axes[1].bar(['Tidak Churn', 'Churn'], after.values, color=['#2E5395', '#4C9F70'])
axes[1].set_title('Sesudah SMOTE (data latih)')
plt.tight_layout(); plt.show()

# Confusion matrix kedua model
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, (nama, model) in zip(axes, [("Baseline", model_base), ("SMOTE", model_smote)]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    ax.imshow(cm, cmap='Blues')
    ax.set_title(nama); ax.set_xlabel('Prediksi'); ax.set_ylabel('Aktual')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Tidak Churn', 'Churn']); ax.set_yticklabels(['Tidak Churn', 'Churn'])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=14, fontweight='bold')
plt.tight_layout(); plt.show()


## ===== 8. Interpretasi & Kesimpulan
