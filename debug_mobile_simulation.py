"""
Mobil landmark cikarmini simule et
Sorun: Android MediaPipe farkli sonuc veriyor olabilir
"""
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.ultra_simple import SimpleLSTM
from src.training.config import NUM_CLASSES, LANDMARK_FEATURES, SEQUENCE_LENGTH

# Model yukle
checkpoint = torch.load('models/best_model.pth', map_location='cpu', weights_only=False)
model = SimpleLSTM(input_size=LANDMARK_FEATURES, num_classes=NUM_CLASSES)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

labels = open('android/app/src/main/assets/labels_tr.txt', 'r', encoding='utf-8').read().strip().split('\n')

print("=" * 60)
print("MOBIL SIMULASYON: Landmark cikarimi sorunlari")
print("=" * 60)

# Senaryo 1: Eller algilanmadi (index 132-258 sifir)
print("\n1. ELLER ALGILANMADI (en yaygin sorun):")
print("   Pose var ama eller 0")
for test in range(3):
    pose_data = np.random.randn(48, 132).astype(np.float32) * 0.3  # Pose landmarks
    hand_data = np.zeros((48, 126), dtype=np.float32)  # Eller sifir
    full_data = np.concatenate([pose_data, hand_data], axis=1)
    
    # Normalize (mobil gibi)
    mean = full_data.mean()
    std = full_data.std() + 1e-8
    normalized = (full_data - mean) / std
    
    input_tensor = torch.from_numpy(normalized).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        top1 = np.argmax(probs)
        print(f"   Test {test+1}: {labels[top1]} ({probs[top1]*100:.1f}%)")

# Senaryo 2: Tek el algilandi (sol veya sag)
print("\n2. TEK EL ALGILANDI:")
print("   Pose + sol el var, sag el 0")
for test in range(3):
    pose_data = np.random.randn(48, 132).astype(np.float32) * 0.3
    left_hand = np.random.randn(48, 63).astype(np.float32) * 0.2
    right_hand = np.zeros((48, 63), dtype=np.float32)
    full_data = np.concatenate([pose_data, left_hand, right_hand], axis=1)
    
    mean = full_data.mean()
    std = full_data.std() + 1e-8
    normalized = (full_data - mean) / std
    
    input_tensor = torch.from_numpy(normalized).unsqueeze(0)
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        top1 = np.argmax(probs)
        print(f"   Test {test+1}: {labels[top1]} ({probs[top1]*100:.1f}%)")

# Senaryo 3: Cok az hareket (motion threshold altinda)
print("\n3. COK AZ HAREKET (ayni frame tekrar ediyor):")
print("   Tum frame'ler neredeyse ayni")
base_frame = np.random.randn(258).astype(np.float32) * 0.3
sequence = np.tile(base_frame, (48, 1))
# Cok kucuk noise ekle
sequence += np.random.randn(48, 258).astype(np.float32) * 0.01

mean = sequence.mean()
std = sequence.std() + 1e-8
normalized = (sequence - mean) / std

input_tensor = torch.from_numpy(normalized).unsqueeze(0)
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0].numpy()
    top3 = np.argsort(probs)[-3:][::-1]
    for idx in top3:
        print(f"   {labels[idx]}: {probs[idx]*100:.1f}%")

# Senaryo 4: Visibility cok dusuk (pose landmark'lari guvensiz)
print("\n4. DUSUK VISIBILITY (pose landmark'lari guvensiz):")
print("   Visibility degerleri 0.1-0.3 arasi")
pose_coords = np.random.randn(48, 99).astype(np.float32) * 0.3  # x,y,z
visibility = np.random.uniform(0.1, 0.3, (48, 33)).astype(np.float32)  # Dusuk visibility
pose_data = np.zeros((48, 132), dtype=np.float32)
for i in range(48):
    for j in range(33):
        pose_data[i, j*4:j*4+3] = pose_coords[i, j*3:j*3+3]
        pose_data[i, j*4+3] = visibility[i, j]

hand_data = np.random.randn(48, 126).astype(np.float32) * 0.2
full_data = np.concatenate([pose_data, hand_data], axis=1)

mean = full_data.mean()
std = full_data.std() + 1e-8
normalized = (full_data - mean) / std

input_tensor = torch.from_numpy(normalized).unsqueeze(0)
with torch.no_grad():
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0].numpy()
    top3 = np.argsort(probs)[-3:][::-1]
    for idx in top3:
        print(f"   {labels[idx]}: {probs[idx]*100:.1f}%")

print("\n" + "=" * 60)
print("COCUK SORUNU ANALIZI:")
print("Eger yukaridaki senaryolarda 'cocuk' sik cikiyorsa:")
print("  -> Mobilde eller duzgun algilanmiyor olabilir")
print("  -> Motion detection cok erken tahmin yapiyor olabilir")
print("  -> Landmark kalitesi dusuk olabilir")
print("=" * 60)
