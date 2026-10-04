# Data Preparation — Studi Kasus Telco Customer Churn

**Mata Kuliah:** Machine Learning — **Kelompok 3**

| Anggota | |
|---|---|
| Frisilia Elfira Maharani | |
| Febyana Mulia Putri | |
| Wanda Nurhasanah | |
| Muhammad Revolvere Rey A | |
| Ahmad Dandi Subhani | |

## Ringkasan

Notebook ini menerapkan alur **Data Preparation** pada dataset nyata **Telco Customer Churn** (7.043 pelanggan, 21 kolom, target `Churn` Yes/No) untuk memprediksi pelanggan yang akan berhenti berlangganan (*churn*).

Tiga masalah data yang ditangani:
1. **Missing values** — diatasi dengan KNN Imputer (fit di data train saja).
2. **Outlier** — dideteksi dan ditangani agar tidak merusak model.
3. **Imbalanced data** — kelas churn minoritas ditangani dengan SMOTE.

Aturan aman yang dipakai: **split dulu (stratified) baru preprocessing**, supaya preprocessing tidak "mengintip" data test. One-hot encoding untuk kolom kategorikal, StandardScaler agar Logistic Regression konvergen.

## Isi Repo

| File | Keterangan |
|---|---|
| `Data_Preparation_Telco_Churn_Kelompok3.ipynb` | Notebook utama (18 cell), siap dibuka di Google Colab lengkap dengan penjelasan tiap langkah |
| `Data_Preparation_Telco_Churn_Kelompok3.py` | Versi script Python dari notebook, enak dibaca/di-copy |
| `Presentasi_Data_Preparation_Kelompok3.pptx` | Slide presentasi (15 slide, termasuk 3 slide Bedah Kode) |
| `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Dataset Telco Customer Churn |

## Cara Menjalankan (Google Colab)

1. Upload `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` ke Colab, atau clone repo ini.
2. Buka notebook `Data_Preparation_Telco_Churn_Kelompok3.ipynb`.
3. Jalankan semua cell berurutan (Runtime > Run all).
4. Cell pertama (`pd.read_csv`) membaca file CSV dari direktori kerja — sesuaikan path bila perlu.

## Alur Notebook (18 cell)

1. Import library (pandas, numpy, scikit-learn, imbalanced-learn, matplotlib)
2. Load dataset Telco Customer Churn
3. Diagnosis masalah data (missing values, outlier, imbalance)
4. Preprocessing: buang `customerID`, encode target, one-hot encoding, split stratified, KNN Imputer, StandardScaler
5. Penanganan imbalanced data (SMOTE)
6. Training & evaluasi model (akurasi, F1-score, confusion matrix)
7. Visualisasi hasil & kesimpulan
