# TID - Türk İşaret Dili Tanıma Sistemi

Gerçek zamanlı Türk İşaret Dili tanıma web uygulaması.

## 🚀 Hızlı Başlangıç

```bash
# tid_old ortamını aktifle
conda activate tid_old

# Web sunucusunu başlat
python app/server.py
```

Tarayıcıda http://localhost:5000 adresine git.

## 📊 Model Bilgileri

- **Model**: `best_model.h5` (TensorFlow/Keras)
- **Doğruluk**: ~63%
- **Sınıf Sayısı**: 226 işaret
- **Input**: 30 frame, 258 landmark feature

## 📁 Proje Yapısı

```
Tid/
├── app/
│   ├── server.py          # Flask web sunucusu
│   ├── templates/         # HTML şablonları
│   └── static/            # CSS/JS dosyaları
├── src/
│   ├── data/              # Veri işleme
│   ├── models/            # Model mimarisi
│   └── training/          # Eğitim scriptleri
├── models/
│   ├── best_model.h5      # Keras model (~63%)
│   └── best_model.pth     # PyTorch model (~73%)
└── AUTSL/                 # Veri seti
```

## 🎯 Kullanım

1. Kameraya doğru işaret yap
2. Tahminler ekranda görünecek
3. "Ekle" ile cümleye kelime ekle
4. "Temizle" ile cümleyi sıfırla

## ⚙️ Gereksinimler

- Python 3.8+
- TensorFlow 2.x
- MediaPipe (eski sürüm, solutions API)
- Flask
- OpenCV

## 📝 Notlar

- `tid_old` conda ortamını kullan (eski MediaPipe için)
- GPU olmadan CPU'da çalışır
- 226 farklı işaret tanınabilir
