"""
DERIN SORUN ANALIZI
Model ve label'lar dogru - sorun gercek zamanli landmark cikarmada
Bu script potansiyel sorunlari tespit eder
"""
import torch
import numpy as np
import sys
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent))  # proje koku
from src.models.ultra_simple import SimpleLSTM
from src.training.config import NUM_CLASSES, LANDMARK_FEATURES, SEQUENCE_LENGTH

import pandas as pd
csv_df = pd.read_csv('AUTSL/SignList_ClassId_TR_EN.csv')
label_map = {row.ClassId: row.TR for _, row in csv_df.iterrows()}

# Model yukle
checkpoint = torch.load('models/best_model.pth', map_location='cpu', weights_only=False)
model = SimpleLSTM(input_size=LANDMARK_FEATURES, num_classes=NUM_CLASSES)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

labels_tr = open('android/app/src/main/assets/labels_tr.txt', 'r', encoding='utf-8').read().strip().split('\n')
labels_tr = [l.strip() for l in labels_tr if l.strip()]

print("=" * 60)
print("DERIN SORUN ANALIZI")
print("=" * 60)

# ============================================
# TEST 1: Val setindeki dogruluk nedir?
# ============================================
print("\n=== TEST 1: Val set dogruluk (50 ornek) ===")
meta = pd.read_csv('processed_data/val/metadata.csv')
val_dir = Path('processed_data/val')

correct = 0
total = 0
wrong_predictions = []

for _, row in meta.sample(n=min(50, len(meta)), random_state=42).iterrows():
    video = row['video']
    actual_label = int(row['label'])
    npy_path = val_dir / f"{video}.npy"
    if not npy_path.exists():
        continue
    
    sample = np.load(str(npy_path)).astype(np.float32)
    mean = sample.mean()
    std = sample.std() + 1e-8
    normalized = ((sample - mean) / std).astype(np.float32)
    t = torch.from_numpy(normalized).unsqueeze(0)
    
    with torch.no_grad():
        pred = model(t)
        probs = torch.softmax(pred, dim=1)[0].numpy()
    
    pred_class = np.argmax(probs)
    if pred_class == actual_label:
        correct += 1
    else:
        wrong_predictions.append({
            'video': video,
            'actual': f"{label_map.get(actual_label,'?')}({actual_label})",
            'predicted': f"{label_map.get(int(pred_class),'?')}({int(pred_class)})",
            'confidence': f"{probs[pred_class]*100:.1f}%",
            'actual_conf': f"{probs[actual_label]*100:.1f}%"
        })
    total += 1

print(f"Dogruluk: {correct}/{total} = {correct/total*100:.1f}%")
print(f"\nYanlis tahminler ({len(wrong_predictions)} adet):")
for wp in wrong_predictions[:10]:
    print(f"  {wp['video']}: Gercek={wp['actual']}, Tahmin={wp['predicted']} (conf={wp['confidence']}, gercek_conf={wp['actual_conf']})")

# ============================================
# TEST 2: LEFT/RIGHT EL SWAP TESTI
# ============================================
print("\n\n=== TEST 2: LEFT/RIGHT EL SWAP TESTI ===")
print("Egitim verisi FLIP yapilmadan isleniyor.")
print("Web/Mobil cv2.flip(frame,1) yapiyor.")
print("Bu flip sol/sag eli TAKAS eder (MediaPipe perspektifinden).\n")

# Bir val ornegi al ve sol/sag elleri takas et
sample_path = list(val_dir.glob('*.npy'))[5]
sample = np.load(str(sample_path)).astype(np.float32)
video_name = sample_path.stem
actual_row = meta[meta.video == video_name]
if len(actual_row) > 0:
    actual_label = int(actual_row.label.values[0])
    
    # Normal tahmin
    mean = sample.mean()
    std = sample.std() + 1e-8
    norm_sample = (sample - mean) / std
    t = torch.from_numpy(norm_sample.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        pred_normal = torch.softmax(model(t), dim=1)[0].numpy()
    
    # Sol/sag elleri takas et (index 132-194 <-> 195-257)
    swapped = sample.copy()
    swapped[:, 132:195] = sample[:, 195:258]  # right -> left slot
    swapped[:, 195:258] = sample[:, 132:195]  # left -> right slot
    
    mean_s = swapped.mean()
    std_s = swapped.std() + 1e-8
    norm_swapped = (swapped - mean_s) / std_s
    t_s = torch.from_numpy(norm_swapped.astype(np.float32)).unsqueeze(0)
    with torch.no_grad():
        pred_swapped = torch.softmax(model(t_s), dim=1)[0].numpy()
    
    normal_top1 = np.argmax(pred_normal)
    swapped_top1 = np.argmax(pred_swapped)
    
    print(f"Video: {video_name}")
    print(f"Gercek: {label_map.get(actual_label, '?')} ({actual_label})")
    print(f"Normal tahmin: {label_map.get(int(normal_top1), '?')} ({int(normal_top1)}, {pred_normal[normal_top1]*100:.1f}%)")
    print(f"Swap tahmin:   {label_map.get(int(swapped_top1), '?')} ({int(swapped_top1)}, {pred_swapped[swapped_top1]*100:.1f}%)")
    print(f"Swap farki yapiyor mu? {'EVET - SWAP SORUN!' if normal_top1 != swapped_top1 else 'Hayir'}")

# Toplu swap testi (20 ornek)
print("\n--- Toplu swap testi (20 ornek) ---")
swap_changes = 0
swap_total = 0
for _, row in meta.sample(n=20, random_state=123).iterrows():
    npy_path = val_dir / f"{row['video']}.npy"
    if not npy_path.exists():
        continue
    
    s = np.load(str(npy_path)).astype(np.float32)
    actual = int(row['label'])
    
    # Normal
    m, st = s.mean(), s.std() + 1e-8
    n_s = (s - m) / st
    with torch.no_grad():
        p1 = model(torch.from_numpy(n_s.astype(np.float32)).unsqueeze(0)).argmax(dim=1).item()
    
    # Swapped
    sw = s.copy()
    sw[:, 132:195], sw[:, 195:258] = s[:, 195:258].copy(), s[:, 132:195].copy()
    m2, st2 = sw.mean(), sw.std() + 1e-8
    n_sw = (sw - m2) / st2
    with torch.no_grad():
        p2 = model(torch.from_numpy(n_sw.astype(np.float32)).unsqueeze(0)).argmax(dim=1).item()
    
    swap_total += 1
    if p1 != p2:
        swap_changes += 1
        # Hangisi dogru?
        status_n = "DOGRU" if p1 == actual else "YANLIS"
        status_s = "DOGRU" if p2 == actual else "YANLIS"
        print(f"  {row['video']}: Normal={label_map.get(p1,'?')}[{status_n}] Swap={label_map.get(p2,'?')}[{status_s}] Gercek={label_map.get(actual,'?')}")

print(f"\nSonuc: {swap_changes}/{swap_total} ornekte swap fark yaratiyor ({swap_changes/max(swap_total,1)*100:.0f}%)")
if swap_changes > swap_total * 0.3:
    print(">>> UYARI: Sol/sag el takasi tahmini onemli olcude etkiliyor!")
    print(">>> Web/Mobil'de yapilan cv2.flip() sol/sag elleri takas ediyor olabilir!")

# ============================================
# TEST 3: EL LANDMARK ORANI
# ============================================
print("\n\n=== TEST 3: EL LANDMARK KALITESI ===")
print("Egitim verisinde ellerin ne kadar algilandigi:\n")

hand_stats = {'both': 0, 'left_only': 0, 'right_only': 0, 'none': 0, 'total': 0}
for _, row in meta.sample(n=100, random_state=55).iterrows():
    npy_path = val_dir / f"{row['video']}.npy"
    if not npy_path.exists():
        continue
    
    s = np.load(str(npy_path)).astype(np.float32)
    hand_stats['total'] += 1
    
    # Her frame icin el durumunu kontrol et
    left_frames = 0
    right_frames = 0
    for frame in s:
        left_sum = np.abs(frame[132:195]).sum()
        right_sum = np.abs(frame[195:258]).sum()
        if left_sum > 0.1:
            left_frames += 1
        if right_sum > 0.1:
            right_frames += 1
    
    left_ratio = left_frames / len(s)
    right_ratio = right_frames / len(s)
    
    if left_ratio > 0.3 and right_ratio > 0.3:
        hand_stats['both'] += 1
    elif left_ratio > 0.3:
        hand_stats['left_only'] += 1
    elif right_ratio > 0.3:
        hand_stats['right_only'] += 1
    else:
        hand_stats['none'] += 1

print(f"  Iki el de algilanan: {hand_stats['both']}/{hand_stats['total']} ({hand_stats['both']/max(hand_stats['total'],1)*100:.0f}%)")
print(f"  Sadece sol el: {hand_stats['left_only']}/{hand_stats['total']} ({hand_stats['left_only']/max(hand_stats['total'],1)*100:.0f}%)")
print(f"  Sadece sag el: {hand_stats['right_only']}/{hand_stats['total']} ({hand_stats['right_only']/max(hand_stats['total'],1)*100:.0f}%)")
print(f"  Hic el yok: {hand_stats['none']}/{hand_stats['total']} ({hand_stats['none']/max(hand_stats['total'],1)*100:.0f}%)")

# ============================================
# TEST 4: CONFIDENCE DAGILIMI
# ============================================
print("\n\n=== TEST 4: CONFIDENCE DAGILIMI ===")
confidences_correct = []
confidences_wrong = []

for _, row in meta.sample(n=100, random_state=77).iterrows():
    npy_path = val_dir / f"{row['video']}.npy"
    if not npy_path.exists():
        continue
    
    s = np.load(str(npy_path)).astype(np.float32)
    actual = int(row['label'])
    m, st = s.mean(), s.std() + 1e-8
    n_s = (s - m) / st
    
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(n_s.astype(np.float32)).unsqueeze(0)), dim=1)[0].numpy()
    
    top1 = np.argmax(probs)
    if top1 == actual:
        confidences_correct.append(probs[top1])
    else:
        confidences_wrong.append(probs[top1])

print(f"Dogru tahminler: {len(confidences_correct)}")
print(f"  Ort confidence: {np.mean(confidences_correct)*100:.1f}%")
print(f"  Min confidence: {np.min(confidences_correct)*100:.1f}%")
print(f"Yanlis tahminler: {len(confidences_wrong)}")
if confidences_wrong:
    print(f"  Ort confidence: {np.mean(confidences_wrong)*100:.1f}%")
    print(f"  Max confidence: {np.max(confidences_wrong)*100:.1f}%")

print("\n" + "=" * 60)
print("OZET")
print("=" * 60)
