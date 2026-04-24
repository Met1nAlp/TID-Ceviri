"""
SWAP DUZELTMESI DOGRULAMA TESTI
Simdi landmark'lar swap edilince dogruluk artmali
"""
import torch
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.ultra_simple import SimpleLSTM
from src.training.config import NUM_CLASSES, LANDMARK_FEATURES, SEQUENCE_LENGTH
import pandas as pd

csv_df = pd.read_csv('AUTSL/SignList_ClassId_TR_EN.csv')
label_map = {row.ClassId: row.TR for _, row in csv_df.iterrows()}

checkpoint = torch.load('models/best_model.pth', map_location='cpu', weights_only=False)
model = SimpleLSTM(input_size=LANDMARK_FEATURES, num_classes=NUM_CLASSES)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

meta = pd.read_csv('processed_data/val/metadata.csv')
val_dir = Path('processed_data/val')

# Test: Normal (egitim verisi, flip yok) vs Swapped (flip simulasyonu)
correct_normal = 0
correct_swapped = 0
total = 0

for _, row in meta.sample(n=100, random_state=42).iterrows():
    npy_path = val_dir / f"{row['video']}.npy"
    if not npy_path.exists():
        continue
    
    s = np.load(str(npy_path)).astype(np.float32)
    actual = int(row['label'])
    
    # Normal (egitim gibi, swap yok)
    m, st = s.mean(), s.std() + 1e-8
    n_s = (s - m) / st
    with torch.no_grad():
        p1 = model(torch.from_numpy(n_s.astype(np.float32)).unsqueeze(0)).argmax(dim=1).item()
    
    # Swapped (flip simulasyonu: Left<->Right takas)
    # Bu, flip yapildiktan sonra swap DUZELTMESI UYGULANMAMIS hali simule eder
    sw = s.copy()
    left_backup = s[:, 132:195].copy()
    sw[:, 132:195] = s[:, 195:258]
    sw[:, 195:258] = left_backup
    m2, st2 = sw.mean(), sw.std() + 1e-8
    n_sw = (sw - m2) / st2
    with torch.no_grad():
        p2 = model(torch.from_numpy(n_sw.astype(np.float32)).unsqueeze(0)).argmax(dim=1).item()
    
    if p1 == actual:
        correct_normal += 1
    if p2 == actual:
        correct_swapped += 1
    total += 1

print("=" * 60)
print("SWAP DUZELTMESI DOGRULAMA")
print("=" * 60)
print(f"\nNormal (egitim verisi, flip yok):     {correct_normal}/{total} = {correct_normal/total*100:.1f}%")
print(f"Swapped (flip yapilip swap UYGULANMAMIS): {correct_swapped}/{total} = {correct_swapped/total*100:.1f}%")
print(f"\nFark: {(correct_normal - correct_swapped)/total*100:.1f} puan")
print()
if correct_normal > correct_swapped:
    print("SONUC: Normal > Swapped")
    print("Bu, swap DUZELTMESININ DOGRU oldugunu gosterir.")
    print("Flip sonrasi el label'larini takas etmek dogruluğu arttirir.")
else:
    print("SONUC: Swapped >= Normal")
    print("Beklenmedik sonuc - daha fazla arastirma gerekli.")
