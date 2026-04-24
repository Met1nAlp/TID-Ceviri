"""
Debug script: Neden her şey 'çocuk' oluyor?
"""
import torch
import numpy as np
import sys
sys.path.append('src')

from models.ultra_simple import SimpleLSTM

# Model yükle
checkpoint = torch.load('models/best_model.pth', map_location='cpu')
model = SimpleLSTM()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Labels
labels = open('android/app/src/main/assets/labels_tr.txt', 'r', encoding='utf-8').read().strip().split('\n')

print("=" * 60)
print("ÇOCUK SORUNU DEBUG")
print("=" * 60)

# Test 1: Sıfır input
print("\n1. Sıfır input (tüm landmark'lar 0):")
zero_input = torch.zeros(1, 48, 258)
with torch.no_grad():
    logits = model(zero_input)
    probs = torch.softmax(logits, dim=1)[0].numpy()
    top3 = np.argsort(probs)[-3:][::-1]
    for idx in top3:
        print(f"   {labels[idx]}: {probs[idx]*100:.1f}%")

# Test 2: Random input (normalize edilmemiş)
print("\n2. Random input (normalize edilmemiş):")
random_input = torch.randn(1, 48, 258)
with torch.no_grad():
    logits = model(random_input)
    probs = torch.softmax(logits, dim=1)[0].numpy()
    top3 = np.argsort(probs)[-3:][::-1]
    for idx in top3:
        print(f"   {labels[idx]}: {probs[idx]*100:.1f}%")

# Test 3: Random input (normalize edilmiş - mobil gibi)
print("\n3. Random input (normalize edilmiş - mobil gibi):")
random_data = np.random.randn(48, 258).astype(np.float32)
mean = random_data.mean()
std = random_data.std() + 1e-8
normalized = (random_data - mean) / std
normalized_input = torch.from_numpy(normalized).unsqueeze(0)
with torch.no_grad():
    logits = model(normalized_input)
    probs = torch.softmax(logits, dim=1)[0].numpy()
    top3 = np.argsort(probs)[-3:][::-1]
    for idx in top3:
        print(f"   {labels[idx]}: {probs[idx]*100:.1f}%")

# Test 4: Çok küçük değerler (landmark algılanmadı)
print("\n4. Çok küçük değerler (landmark algılanmadı):")
small_input = torch.ones(1, 48, 258) * 0.001
with torch.no_grad():
    logits = model(small_input)
    probs = torch.softmax(logits, dim=1)[0].numpy()
    top3 = np.argsort(probs)[-3:][::-1]
    for idx in top3:
        print(f"   {labels[idx]}: {probs[idx]*100:.1f}%")

# Test 5: Sadece pose var, eller yok (index 132-258 sıfır)
print("\n5. Sadece pose var, eller yok:")
pose_only = torch.randn(1, 48, 258)
pose_only[:, :, 132:] = 0  # Elleri sıfırla
with torch.no_grad():
    logits = model(pose_only)
    probs = torch.softmax(logits, dim=1)[0].numpy()
    top3 = np.argsort(probs)[-3:][::-1]
    for idx in top3:
        print(f"   {labels[idx]}: {probs[idx]*100:.1f}%")

# Test 6: Sadece eller var, pose yok
print("\n6. Sadece eller var, pose yok:")
hands_only = torch.zeros(1, 48, 258)
hands_only[:, :, 132:] = torch.randn(1, 48, 126)
with torch.no_grad():
    logits = model(hands_only)
    probs = torch.softmax(logits, dim=1)[0].numpy()
    top3 = np.argsort(probs)[-3:][::-1]
    for idx in top3:
        print(f"   {labels[idx]}: {probs[idx]*100:.1f}%")

print("\n" + "=" * 60)
print("SONUÇ:")
print("Eğer yukarıdaki testlerin çoğunda 'çocuk' çıkıyorsa,")
print("model kötü input'larda default olarak 'çocuk' tahmin ediyor.")
print("Bu durumda mobilde landmark çıkarımı hatalı olabilir.")
print("=" * 60)
