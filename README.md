# AI Prediksi Gaji IT Indonesia — Streamlit + Random Forest

Aplikasi **Streamlit** untuk memprediksi estimasi gaji IT di Indonesia berbasis Machine Learning (**Random Forest**) dan menampilkan analisis pasar (median gaji per lokasi, tren pengalaman, top tech stack, dll).

> Proyek ini dibuat untuk tujuan edukasi/referensi. Hasil prediksi bersifat estimasi dan dapat berbeda dari kondisi pasar nyata.

---

## Demo
- Local: `http://localhost:8501`
- Deploy (Streamlit Community Cloud): `https://<app-name>.streamlit.app` (isi setelah kamu deploy)

---

## Fitur Utama
- Prediksi gaji berdasarkan:
  - Role/Posisi
  - Lokasi kerja
  - Tech stack utama
  - Pengalaman kerja (tahun atau dari tanggal mulai kerja)
- Analisis:
  - Persentil gaji (posisi kamu dibanding populasi dataset)
  - Range gaji wajar (confidence interval)
  - Grafik radar (dampak faktor role/lokasi/tech/pengalaman)
  - Grafik perbandingan median gaji berdasarkan lokasi
  - Tren gaji vs pengalaman (mean & median)
- Riwayat prediksi:
  - Tabel riwayat + download CSV
- Export:
  - Download laporan prediksi dalam JSON

---

## Teknologi
- Python, Streamlit
- pandas, numpy
- scikit-learn (RandomForestRegressor, Pipeline, OneHotEncoder)
- plotly (visualisasi)

---

## Struktur Repository (Disarankan)
Streamlit Community Cloud mengeksekusi app dari root repository, jadi jalankan lokal juga dari root agar path konsisten [web:32].

```txt
ai-prediksi-gaji/
├─ app.py
├─ requirements.txt
├─ (optional) Indonesia_AI_Data_Salary_Dataset.csv
└─ .streamlit/
   ├─ config.toml        # optional
   └─ secrets.toml       # optional (untuk lokal)
