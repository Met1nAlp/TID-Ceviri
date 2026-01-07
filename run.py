"""
Quick start script for TID Recognition System
Run all steps in sequence
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{'='*60}")
    print(f"⏳ {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(command, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed")
        return False
    
    print(f"\n✓ {description} completed successfully")
    return True


def main():
    print("\n" + "="*60)
    print("🚀 TID Recognition System - Quick Start")
    print("="*60)
    
    # Check Python
    print(f"\nPython: {sys.version}")
    
    # Check CUDA
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("⚠ PyTorch not installed. Run: pip install -r requirements.txt")
        return
    
    print("\n" + "="*60)
    print("Select an option:")
    print("="*60)
    print("1. Data Preprocessing (extract landmarks)")
    print("2. Train Model")
    print("3. Run Real-time Recognition (Desktop)")
    print("4. Run Web Application")
    print("5. All steps (1 → 2 → 4)")
    print("0. Exit")
    
    choice = input("\nEnter choice (0-5): ").strip()
    
    if choice == "1":
        run_command("python src/data/preprocess.py", 
                   "Data Preprocessing")
    
    elif choice == "2":
        run_command("python src/training/train.py --model landmark_only", 
                   "Model Training")
    
    elif choice == "3":
        run_command("python src/inference/realtime.py", 
                   "Real-time Recognition")
    
    elif choice == "4":
        print("\n🌐 Starting web server...")
        print("Open http://localhost:5000 in your browser")
        run_command("python app/server.py", 
                   "Web Application")
    
    elif choice == "5":
        if run_command("python src/data/preprocess.py", 
                      "Data Preprocessing"):
            if run_command("python src/training/train.py --model landmark_only", 
                          "Model Training"):
                print("\n🌐 Starting web server...")
                print("Open http://localhost:5000 in your browser")
                run_command("python app/server.py", 
                           "Web Application")
    
    elif choice == "0":
        print("\nGoodbye! 👋")
    
    else:
        print("\n❌ Invalid choice")


if __name__ == "__main__":
    main()
