# Panduan Deployment Streamlit Cloud - Eye Disease Classification

Panduan ini akan membantu Anda mengupload project ke GitHub dan mendeploynya ke Streamlit Cloud.

## 1. Persiapan Repository GitHub

Streamlit Cloud membutuhkan code Anda berada di repository GitHub (Public atau Private).

### Langkah 1: Buat Repository Baru
1. Login ke [GitHub](https://github.com).
2. Buat repository baru (tombol **New**).
3. Beri nama, misalnya `eye-disease-streamlit`.
4. **JANGAN** centang "Add a README file" (kita akan push dari lokal).
5. Klik **Create repository**.

### Langkah 2: Upload File dari Komputer Lokal
Di folder project Anda (`d:\Kuliah\5th-Semester\Project\EyeDiseaseClassification`), buka terminal (Command Prompt/PowerShell/Git Bash) dan jalankan perintah berikut:

```bash
# Inisialisasi git jika belum ada
git init

# Tambahkan semua file
git add .

# Commit pertama
git commit -m "Initial commit for Streamlit Cloud"

# Ubah branch ke main (opsional tapi disarankan)
git branch -M main

# Hubungkan ke repo GitHub yang barusan dibuat
# GANTI URL DI BAWAH dengan URL repo Anda (misal: https://github.com/username/eye-disease-streamlit.git)
git remote add origin https://github.com/USERNAME/NAMA-REPO-ANDA.git

# Push ke GitHub
git push -u origin main
```

> **Catatan Git LFS (Large File Storage):**
> File `mobilenet_fundus.keras` Anda berukuran sekitar 9-10MB, yang mana **aman** untuk GitHub biasa (batasnya 100MB). Anda **TIDAK PERLU** Git LFS untuk file ini.

---

## 2. Deployment ke Streamlit Cloud

### Langkah 1: Login ke Streamlit
1. Buka [share.streamlit.io](https://share.streamlit.io/).
2. Login menggunakan akun GitHub Anda.

### Langkah 2: Buat App Baru
1. Klik tombol **New app** (biasanya di pojok kanan atas).
2. Pilih **Use existing repo**.

### Langkah 3: Konfigurasi App
Isi form dengan detail berikut:
- **Repository**: Pilih repo yang baru Anda buat (misal: `username/eye-disease-streamlit`).
- **Branch**: `main` (atau `master` tergantung yang Anda pakai).
- **Main file path**: `app.py` (biarkan default jika file ada di root).
- **App URL**: (Opsional) Anda bisa kustomisasi subdomain.

### Langkah 4: Deploy!
1. Klik tombol **Deploy!**.
2. Tunggu proses "baking" (instalasi requirements) selesai. Ini mungkin memakan waktu 1-3 menit.
3. Jika sukses, aplikasi Anda akan terbuka dan siap digunakan!

---

## 3. Maintenance & Update

Jika Anda mengubah kode atau model di masa depan:

1. Edit file di komputer lokal.
2. Commit dan push lagi:
   ```bash
   git add .
   git commit -m "Update model/kode"
   git push
   ```
3. Streamlit Cloud akan mendeteksi perubahan dan otomatis me-redeploy aplikasi Anda (biasanya sangat cepat).

---

## Troubleshooting Umum

- **Error: `ModuleNotFoundError`**: Pastikan library tersebut ada di `requirements.txt`.
- **Error: `FileNotFoundError`**: Pastikan path file benar. Di Streamlit Cloud, struktur folder sama persis dengan yang ada di GitHub.
- **App Crash saat Load Model**: Biasanya karena memory limit. MobileNetV2 cukup ringan, tapi jika terjadi, coba reboot app dari dashboard Streamlit.
