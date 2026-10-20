# 05 — Human-Computer Interaction & Data Visualization

## Deskripsi & Tujuan

Modul ini mengimplementasikan dashboard analitik interaktif berbasis web menggunakan **Streamlit** dan visualisasi responsif dengan **Plotly**. Dashboard ini dirancang untuk mendukung interaksi manusia dan komputer (*Human-Computer Interaction* - HCI) dalam menelaah, memvalidasi, dan menginterpretasikan hasil prediksi kualitas udara (Indeks Standar Pencemar Udara / ISPU).

Prinsip HCI & Visualisasi yang diterapkan:
1. **Hierarki Visual & Information Chunking**: Metrik utama (MAE, RMSE, R², data count) ditempatkan di bagian atas (*KPI cards*) dengan badge indikator performa yang mudah dipahami.
2. **Fleksibilitas Input Data (Dual-Mode)**:
   - **Mode 1 (Otomatis)**: Mendeteksi file `predictions_test.csv` yang dihasilkan oleh pipeline Machine Learning di `04-machine-learning-ispu/output/`.
   - **Mode 2 (Unggah Berkas / CSV Upload)**: Pengguna dapat mengunggah file CSV hasil prediksinya sendiri dengan sistem validasi kolom otomatis.
   - **Mode 3 (Data Contoh Sintetis)**: Menggunakan data sampel bawaan jika belum ada file model yang dijalankan.
3. **Analisis Komparatif Multi-Model**: Membandingkan performa model Random Forest vs ANFIS-PSO secara simultan.
4. **Analisis Residu & Distribusi Kesalahan**: Menyajikan histogram residual ($y_{aktual} - y_{prediksi}$) dengan kurva normal dan garis nol untuk memeriksa adanya bias sistematis model (*systematic over/under-prediction*).
5. **Transparansi & Peringatan Etika AI**: Banner peringatan yang jelas dan catatan keterbatasan data buatan komputer.

---

## Struktur Folder

```
05-hci-data-visualization/
├── app.py                            # Aplikasi dashboard interaktif Streamlit
├── data/
│   └── hasil_prediksi_sample.csv     # Data sampel hasil prediksi
├── tests/
│   └── test_dashboard.py             # Unit tests (pytest)
├── requirements.txt                  # Dependensi Python
└── README.md                         # Dokumentasi panduan
```

---

## Instalasi & Menjalankan Dashboard

### 1. Menyiapkan Virtual Environment

```bash
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Menjalankan Dashboard Web

```bash
streamlit run app.py
```

Setelah perintah dijalankan, dashboard akan otomatis terbuka di peramban web pada alamat `http://localhost:8501`.

### 3. Menjalankan Pengujian Otomatis (Unit Tests)

```bash
pytest tests/ -v
```

Unit tests memverifikasi fungsi validasi skema data, penanganan kolom hilang/tidak valid, toleransi data kosong, dan kalkulasi metrik tanpa perlu me-render antarmuka grafis browser.

---

## Format File Input untuk Mode Upload

Jika mengunggah file CSV sendiri, pastikan file memiliki kolom-kolom berikut:
- `tanggal` (format tanggal `YYYY-MM-DD`)
- `ispu_aktual` (angka riil/float nilai target observasi)
- Kolom prediksi: salah satu atau kedua kolom `prediksi_rf` dan/atau `prediksi_anfis_pso`.

---

## Catatan Integritas & Validitas Ilmiah

> [!WARNING]
> Nilai metrik evaluasi (MAE, RMSE, $R^2$) yang ditampilkan di dashboard dihitung secara dinamis pada rentang tanggal yang dipilih oleh pengguna di antarmuka. Angka ini bertujuan membantu eksplorasi data secara visual dan **bukan pengganti metrik evaluasi formal** yang dihitung pada data uji terisolasi (*held-out test set*) dalam naskah skripsi/publikasi ilmiah.
