"""
Cocuk sinifinin ozelliklerini analiz et
Neden her sey cocuk oluyor?
"""
import torch
import numpy as np
import sys
sys.path.append('src')

from models.ultra_simple import SimpleLSTM

checkpoint = torch.load('models/best_model.pth', map_location='cpu')
model = SimpleLSTM()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

labels = open('android/app/src/main/assets/labels_tr.txt', 'r', encoding='utf-8').read().strip().split('\n')

print("=" * 60)
print("COCUK SINIFI ANALIZI")
print("=" * 60)

# Egitim verisinden cocuk orneklerini yukle
import os
processed_dir = 'processed_data/train'
cocuk_files = [f for f in os.listdir(processed_dir) if f.endswith('_42.npy')]

if len(cocuk_files) > 0:
    print(f"\nCocuk sinifi ornek sayisi: {len(cocuk_files)}")
    
    # Ilk 5 ornegi yukle ve tahmin yap
    print("\nEgitim verisindeki cocuk ornekleri:")
    for i, fname in enumerate(cocuk_files[:5]):
        data = np.load(os.path.join(processed_dir, fname))
        
        # Normalize (egitim gibi)
        mean = data.mean()
        std = data.std() + 1e-8
        normalized = (data - mean) / std
        
        input_tensor = torch.from_numpy(normalized).unsqueeze(0)
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)[0].numpy()
            top3 = np.argsort(probs)[-3:][::-1]
            
            print(f"\n  Ornek {i+1} ({fname}):")
            for idx in top3:
                marker = " <-- DOGRU" if idx == 42 else ""
                print(f"    {labels[idx]}: {probs[idx]*100:.1f}%{marker}")
    
    # Cocuk orneklerinin ortalama landmark istatistikleri
    print("\n" + "=" * 60)
    print("COCUK ORNEKLERININ OZELLIKLERI:")
    
    all_cocuk = []
    for fname in cocuk_files[:20]:
        data = np.load(os.path.join(processed_dir, fname))
        all_cocuk.append(data)
    
    all_cocuk = np.array(all_cocuk)
    
    print(f"\nPose landmarks (0-131):")
    print(f"  Ortalama: {all_cocuk[:, :, :132].mean():.4f}")
    print(f"  Std: {all_cocuk[:, :, :132].std():.4f}")
    
    print(f"\nSol el landmarks (132-194):")
    print(f"  Ortalama: {all_cocuk[:, :, 132:195].mean():.4f}")
    print(f"  Std: {all_cocuk[:, :, 132:195].std():.4f}")
    print(f"  Sifir oran: {(all_cocuk[:, :, 132:195] == 0).mean()*100:.1f}%")
    
    print(f"\nSag el landmarks (195-257):")
    print(f"  Ortalama: {all_cocuk[:, :, 195:].mean():.4f}")
    print(f"  Std: {all_cocuk[:, :, 195:].std():.4f}")
    print(f"  Sifir oran: {(all_cocuk[:, :, 195:] == 0).mean()*100:.1f}%")
    
else:
    print("\nCocuk ornekleri bulunamadi!")
    print("processed_data/train/ dizinini kontrol et")

print("\n" + "=" * 60)
