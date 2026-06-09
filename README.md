# DeepSign-TID — Turk Isaret Dili Tanima Sistemi

Konya Teknik Universitesi - Bilgisayar Muhendisligi Bolumu  
Bitirme Projesi, Haziran 2026

**Ogrenci:** Metin KAYAALP  
**Danisaman:** Prof. Dr. Ahmet BABALIK

---

## Proje Hakkinda

DeepSign-TID, gercek zamanli Turk Isaret Dili (TID) tanima sistemidir. Kamera goruntulerinden el ve vucut hareketlerini algilayarak 226 farkli isareti taniyabilir, metne donusturebilir ve seslendirebilir. Sistem hem web tarayici uzerinden hem de Android mobil uygulama olarak calisabilmektedir.

---

## Sistem Gereksinimleri

| Gereksinim | Minimum |
|------------|---------|
| Isletim Sistemi | Windows 10/11 |
| Python | 3.10 veya ustu |
| RAM | 8 GB |
| GPU (opsiyonel) | CUDA destekli NVIDIA GPU (egitim icin onerilir) |
| Kamera | Web kamerasi (canli tanima icin) |
| Disk Alani | ~2 GB (veri seti haric) |

---

## Kurulum

### 1. Python Ortaminin Hazirlanmasi

Anaconda veya standart Python kurulumu kullanilabilir.

**Anaconda ile (onerilen):**

```bash
conda create -n deepsign python=3.13
conda activate deepsign
```

**Standart Python ile:**

```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Bagimliliklarin Kurulmasi

```bash
pip install -r requirements.txt
```

PyTorch GPU destegi isteniyorsa (NVIDIA GPU icin):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

GPU yoksa CPU surumu otomatik olarak kurulacaktir, sistem CPU ile de calisir.

### 3. Veri Seti (Sadece Egitim Icin Gerekli)

AUTSL veri seti asagidaki adresten indirilebilir:

- Kaynak: https://cvml.ankara.edu.tr/datasets/
- Indirilen videolar `AUTSL/train/`, `AUTSL/val/`, `AUTSL/test/` klasorlerine yerlestirilmelidir.

**Not:** Veri seti yalnizca modeli sifirdan egitmek icin gereklidir. Onceden egitilmis model dosyalari (`models/` klasoru) projeye dahil edilmistir; dogrudan calistirmak icin veri setine gerek yoktur.

---

## Calistirma

### Web Uygulamasi (Onerilen)

```bash
python run.py
```

Acilan menuden **4 numarali secenegi** (Web Application) secin. Tarayicinizda asagidaki adresi acin:

```
http://localhost:5000
```

Kameraniza dogru isaret yapin. Sistem hareketi otomatik olarak algilayacak, tahmini ekranda gosterecektir.

### Diger Secenekler

`run.py` icerisindeki menu:

| Secenek | Aciklama |
|---------|----------|
| 1 | Veri on isleme (MediaPipe ile anahtar nokta cikarimi) |
| 2 | Model egitimi |
| 3 | Masaustu gercek zamanli tanima (OpenCV penceresi) |
| 4 | Web uygulamasi (tarayici uzerinden) |
| 5 | Tum adimlari sirayla calistir (1 > 2 > 4) |

---

## Proje Yapisi

```
DeepSign-TID/
|
|-- src/                          # Kaynak kod
|   |-- data/
|   |   |-- preprocess.py         # MediaPipe ile videodan anahtar nokta cikarimi
|   |   |-- dataset.py            # PyTorch Dataset sinifi ve veri artirma
|   |-- models/
|   |   |-- ultra_simple.py       # MLP ve BiLSTM model tanimlari
|   |   |-- hybrid_model.py       # Hibrit model (GRU + CNN)
|   |-- training/
|   |   |-- train.py              # Model egitim scripti
|   |   |-- config.py             # Hiperparametre ayarlari
|   |   |-- focus.py              # Focal loss fonksiyonu
|   |-- inference/
|   |   |-- realtime.py           # Gercek zamanli cikarim ve kayan pencere yontemi
|   |-- digit_selection/          # Parmak secimi alt modulu
|
|-- app/                          # Web uygulamasi
|   |-- server.py                 # Flask web sunucusu
|   |-- pytorch_predictor.py      # TID tahmin motoru (MediaPipe + BiLSTM)
|   |-- digit_selection_predictor.py  # Parmak secimi tahmin motoru
|   |-- templates/index.html      # Web arayuzu
|   |-- static/                   # CSS ve JavaScript dosyalari
|
|-- android/                      # Android mobil uygulama (Kotlin/Jetpack Compose)
|
|-- models/                       # Egitilmis model dosyalari
|   |-- best_model.pth            # Ana BiLSTM modeli (~34 MB, %78.36 dogruluk)
|   |-- best_model_mobile.ptl     # Mobil format (PyTorch Lite, ~11 MB)
|   |-- digit_selection_best.pth  # Parmak secimi modeli
|   |-- digit_selection_mobile.ptl # Parmak secimi mobil modeli
|
|-- AUTSL/                        # AUTSL veri seti (train/val/test videolari ve CSV etiketleri)
|-- processed_data/               # On islemden gecmis anahtar nokta verileri
|-- external_data/                # Ek veri seti (isaret dili rakam veri seti)
|
|-- run.py                        # Ana baslatici
|-- requirements.txt              # Python bagimliliklari
|-- benchmark_live_pipeline.py    # Canli pipeline basarim testi
|-- analyze_benchmark_report.py   # Benchmark sonuc analiz scripti
|-- export_mobile_model.py        # Modeli mobil formata donusturme
|-- training_plot.png             # Egitim grafigi (dogruluk ve kayip egrileri)
|-- training_mobile_plot.png      # Mobil model egitim grafigi
```

---

## Teknik Detaylar

### Model Mimarisi

- 2 katmanli cift yonlu LSTM (BiLSTM)
- Attention pooling (tum zaman adimlarinda agirlikli ortalama)
- Katman normallestirme (LayerNorm) ve seyreltme (Dropout 0.5)
- Yaklasik 2.8 milyon parametre

### Veri Isleme

- Veri seti: AUTSL (226 sinif, yaklasik 38.000 video)
- Her video karesinden MediaPipe ile 258 boyutlu anahtar nokta vektoru cikarilir
  - 33 vucut noktasi x 4 koordinat + 21 sol el x 3 + 21 sag el x 3
- Zaman ekseni 48 kare olarak sabitlenir (~1.6 saniye)

### Mobil Entegrasyon

- Model PyTorch Lite formatina (~10.9 MB) donusturulmustur
- Android uygulamasi Kotlin ve Jetpack Compose ile gelistirilmistir
- CameraX ile goruntu alimi, Android TTS ile seslendirme yapilir
- Sunucu bagimliligi olmadan cihaz uzerinde calisir

---

## Sonuclar

| Olcut | Dogrulama | Canli Benchmark |
|-------|-----------|-----------------|
| Ilk-1 Dogruluk | %78.36 | %50.44 |
| Ilk-3 Dogruluk | ~%90+ | %69.03 |
| Kapsama Orani | - | %100 |

- 226 sinif uzerinde %78.36 dogruluk elde edilmistir.
- Ilk-3 dogruluk %91.40 ile neredeyse her zaman dogru tahmini icermektedir.
- Canli benchmark, 452 video uzerinde gercek kosullarda degerlendirilmistir.

---

## Sik Karsilasilan Sorunlar

| Sorun | Cozum |
|-------|-------|
| `ModuleNotFoundError: No module named 'torch'` | `pip install -r requirements.txt` komutunu calistirin |
| Kamera acilmiyor | Baska bir uygulama kamerayi kullaniyor olabilir, kapatin ve tekrar deneyin |
| CUDA hatasi | GPU suruculerinizi guncelleyin veya CPU modunda calistirin (GPU zorunlu degildir) |
| Web sayfasi acilmiyor | `http://localhost:5000` adresini kullandiginizdan emin olun |
| Model dosyasi bulunamadi | `models/` klasorunun proje dizininde oldugunu kontrol edin |

---

## Lisans

MIT License

## Kaynaklar

- AUTSL Dataset - Ankara Universitesi (https://cvml.ankara.edu.tr/datasets/)
- MediaPipe - Google (https://mediapipe.dev/)
- PyTorch (https://pytorch.org/)
