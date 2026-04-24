"""
Egitim kalitesi ve landmark kalitesi analizi
"""
import numpy as np
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("EGITIM VE LANDMARK KALITE ANALIZI")
print("=" * 70)

# 1. PROCESSED DATA KALITE KONTROLU
print("\n1. LANDMARK KALITE KONTROLU (processed_data)")
print("-" * 70)

train_dir = 'processed_data/train'
if os.path.exists(train_dir):
    files = [f for f in os.listdir(train_dir) if f.endswith('.npy')][:100]
    
    all_data = []
    zero_hand_count = 0
    low_visibility_count = 0
    
    for fname in files:
        data = np.load(os.path.join(train_dir, fname))
        all_data.append(data)
        
        # El kontrolu (index 132-258)
        hands = data[:, 132:]
        if np.all(hands == 0):
            zero_hand_count += 1
        
        # Visibility kontrolu (pose landmark'lari, her 4. deger)
        visibility = data[:, 3::4][:, :33]  # Sadece pose visibility
        if visibility.mean() < 0.5:
            low_visibility_count += 1
    
    all_data = np.array(all_data)
    
    print(f"Analiz edilen ornek: {len(files)}")
    print(f"\nPOSE LANDMARKS (0-131):")
    print(f"  Ortalama: {all_data[:, :, :132].mean():.4f}")
    print(f"  Std: {all_data[:, :, :132].std():.4f}")
    print(f"  Min: {all_data[:, :, :132].min():.4f}")
    print(f"  Max: {all_data[:, :, :132].max():.4f}")
    
    print(f"\nSOL EL LANDMARKS (132-194):")
    print(f"  Ortalama: {all_data[:, :, 132:195].mean():.4f}")
    print(f"  Std: {all_data[:, :, 132:195].std():.4f}")
    print(f"  Sifir oran: {(all_data[:, :, 132:195] == 0).mean()*100:.1f}%")
    
    print(f"\nSAG EL LANDMARKS (195-257):")
    print(f"  Ortalama: {all_data[:, :, 195:].mean():.4f}")
    print(f"  Std: {all_data[:, :, 195:].std():.4f}")
    print(f"  Sifir oran: {(all_data[:, :, 195:] == 0).mean()*100:.1f}%")
    
    print(f"\nKALITE METRIKLERI:")
    print(f"  Hic el olmayan ornek: {zero_hand_count}/{len(files)} ({zero_hand_count/len(files)*100:.1f}%)")
    print(f"  Dusuk visibility: {low_visibility_count}/{len(files)} ({low_visibility_count/len(files)*100:.1f}%)")
    
    # SORUN TESPITI
    print(f"\n{'='*70}")
    print("SORUN TESPITI:")
    
    hand_zero_ratio = (all_data[:, :, 132:] == 0).mean()
    if hand_zero_ratio > 0.3:
        print(f"  [KRITIK] El landmark'lari %{hand_zero_ratio*100:.0f} sifir!")
        print(f"           -> Egitim verisinde eller duzgun algilanmamis")
        print(f"           -> Landmark'lari YENIDEN CIKAR")
    elif hand_zero_ratio > 0.15:
        print(f"  [UYARI] El landmark'lari %{hand_zero_ratio*100:.0f} sifir")
        print(f"          -> Bazi videolarda eller algilanmamis")
    else:
        print(f"  [OK] El landmark kalitesi iyi (%{hand_zero_ratio*100:.0f} sifir)")
    
    if zero_hand_count > len(files) * 0.05:
        print(f"  [KRITIK] %{zero_hand_count/len(files)*100:.0f} ornekte HIC el yok!")
        print(f"           -> MediaPipe parametrelerini kontrol et")
        print(f"           -> Landmark'lari YENIDEN CIKAR")
    
    visibility_mean = all_data[:, :, 3::4][:, :, :33].mean()
    if visibility_mean < 0.7:
        print(f"  [UYARI] Ortalama visibility dusuk: {visibility_mean:.2f}")
        print(f"          -> Video kalitesi dusuk olabilir")
    else:
        print(f"  [OK] Visibility iyi: {visibility_mean:.2f}")

else:
    print("processed_data/train/ bulunamadi!")

# 2. MODEL PERFORMANS ANALIZI
print(f"\n{'='*70}")
print("2. MODEL PERFORMANS ANALIZI")
print("-" * 70)

import torch
from src.models.ultra_simple import SimpleLSTM
from src.training.config import NUM_CLASSES, LANDMARK_FEATURES

checkpoint = torch.load('models/best_model.pth', map_location='cpu', weights_only=False)
model = SimpleLSTM(input_size=LANDMARK_FEATURES, num_classes=NUM_CLASSES)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model: SimpleLSTM")
print(f"Parametreler: {sum(p.numel() for p in model.parameters()):,}")
print(f"Epoch: {checkpoint.get('epoch', 'N/A')}")
print(f"Val Accuracy: {checkpoint.get('best_val_acc', 0):.2f}%")

# Top-3 accuracy varsa
if 'top3_val_acc' in checkpoint:
    print(f"Top-3 Val Accuracy: {checkpoint.get('top3_val_acc', 0):.2f}%")

# 3. SINIF BAZLI PERFORMANS (varsa)
if 'class_accuracies' in checkpoint:
    print(f"\n3. SINIF BAZLI PERFORMANS")
    print("-" * 70)
    class_accs = checkpoint['class_accuracies']
    labels = open('android/app/src/main/assets/labels_tr.txt', 'r', encoding='utf-8').read().strip().split('\n')
    
    # En kotu 10 sinif
    worst_classes = sorted(enumerate(class_accs), key=lambda x: x[1])[:10]
    print("\nEN KOTU 10 SINIF:")
    for idx, acc in worst_classes:
        print(f"  {labels[idx]}: {acc:.1f}%")
    
    # Cocuk sinifi
    if len(class_accs) > 42:
        print(f"\nCOCUK SINIFI:")
        print(f"  Accuracy: {class_accs[42]:.1f}%")
        if class_accs[42] < 50:
            print(f"  [UYARI] Cocuk sinifi kotu ogrenilmis!")

print(f"\n{'='*70}")
print("GENEL DEGERLENDIRME:")
print("-" * 70)

val_acc = checkpoint.get('best_val_acc', 0)
if val_acc > 75:
    print(f"[OK] Model performansi iyi: {val_acc:.1f}%")
elif val_acc > 60:
    print(f"[ORTA] Model performansi orta: {val_acc:.1f}%")
    print("      -> Daha fazla epoch egit")
    print("      -> Augmentation arttir")
else:
    print(f"[KOTU] Model performansi dusuk: {val_acc:.1f}%")
    print("       -> Landmark kalitesini kontrol et")
    print("       -> Model mimarisini degistir")

print(f"\n{'='*70}")
print("TAVSIYE:")
print("-" * 70)

hand_zero_ratio = (all_data[:, :, 132:] == 0).mean() if 'all_data' in locals() else 0
if hand_zero_ratio > 0.3 or zero_hand_count > len(files) * 0.05:
    print(">>> LANDMARK'LARI YENIDEN CIKAR <<<")
    print("    Komut: python src/data/preprocess.py")
    print("    Neden: El landmark'lari kalitesiz")
elif val_acc < 70:
    print(">>> MODELI YENIDEN EGIT <<<")
    print("    Komut: python src/training/train.py --model lstm --epochs 100")
    print("    Neden: Model performansi dusuk")
else:
    print(">>> EGITIM VE LANDMARK KALITESI IYI <<<")
    print("    Mobil taraftaki duzeltmeler yeterli olmali")
    print("    Uygulamayi test et ve sonuclari gozlemle")

print(f"{'='*70}\n")
