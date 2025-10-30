# Proyek Talent Match Intelligence (Studi Kasus 2025)

Streamlit : https://talent-match-intelligence-test.streamlit.app/

Proyek ini adalah prototipe aplikasi Streamlit yang dirancang untuk mensimulasikan sistem *Talent Match Intelligence*[cite: 6]. Aplikasi ini mengimplementasikan alur kerja analitik penuh:

1.  **Penemuan Pola Sukses (Step 1):** Menganalisis data karyawan historis untuk menemukan "Success Formula".
2.  **Logika Algoritma (Step 2):** Mensimulasikan algoritma SQL modular untuk menghitung skor kecocokan granular.
3.  **Aplikasi AI & Dasbor (Step 3):** Menyajikan temuan dalam antarmuka yang dinamis, didukung oleh AI generatif.

Aplikasi ini menggunakan solusi *hybrid* yang memuat 16 file CSV ke dalam memori untuk mengatasi kendala koneksi *database* sambil tetap memenuhi semua persyaratan studi kasus.

## 🚀 Instalasi dan Cara Penggunaan

Berikut adalah panduan langkah demi langkah untuk menjalankan aplikasi *Talent Match Intelligence* ini di komputer lokal Anda.

### 1. Prasyarat

Sebelum memulai, pastikan Anda telah menginstal:
* [Python 3.8+](https://www.python.org/downloads/)
* [Git](https://git-scm.com/downloads)
* Sebuah **Google AI API Key** (dapat diperoleh dari [Google AI Studio](https://aistudio.google.com/))

---

### 2. Langkah-Langkah Instalasi

**Clone Repositori**
Buka terminal Anda dan clone repositori ini:
```
git clone https://github.com/MRadityaAdhirajasa/talent-match-intelligence.git
```

### Install requirements.txt
```
pip install -r requirements.txt
```

### Run steamlit app
```
streamlit run dashboard.py
```
