"""
PyTorch Mobile Model Export Script
Converts best_model.pth to mobile-optimized .ptl format
Run: python export_mobile_model.py
"""
import torch
import sys
sys.path.append('src')

from models.ultra_simple import SimpleLSTM
from training.config import NUM_CLASSES, LANDMARK_FEATURES, SEQUENCE_LENGTH

print("Loading model...")
checkpoint = torch.load('models/best_model.pth', map_location='cpu')
model = SimpleLSTM(
    input_size=LANDMARK_FEATURES,
    num_classes=NUM_CLASSES
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Model loaded! Val accuracy: {checkpoint.get('best_val_acc', 'N/A'):.2f}%")

# Trace the model with a sample input
sample_input = torch.zeros(1, SEQUENCE_LENGTH, LANDMARK_FEATURES)
print(f"Tracing with input shape: {sample_input.shape}")

with torch.no_grad():
    traced = torch.jit.trace(model, sample_input)

# Optimize for mobile
from torch.utils.mobile_optimizer import optimize_for_mobile
optimized = optimize_for_mobile(traced)

# Save
output_path = 'models/best_model_mobile.ptl'
optimized._save_for_lite_interpreter(output_path)

import os
size_mb = os.path.getsize(output_path) / 1024 / 1024
print(f"\n✅ Mobile model saved: {output_path}")
print(f"   Size: {size_mb:.1f} MB")
print(f"   Input: (1, {SEQUENCE_LENGTH}, {LANDMARK_FEATURES})")
print(f"   Output: (1, {NUM_CLASSES}) classes")
print(f"\nNow copy 'models/best_model_mobile.ptl' to Android assets folder:")
print(f"   android/app/src/main/assets/best_model_mobile.ptl")
