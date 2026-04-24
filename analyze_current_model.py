# -*- coding: utf-8 -*-
"""
Mevcut model durumu tam analiz scripti.
Calistirma: python analyze_current_model.py
"""
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.path.append(os.path.dirname(__file__))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import defaultdict

from src.training.config import (
    DEVICE, NUM_CLASSES, LANDMARK_FEATURES, MODEL_DIR
)
from src.models.ultra_simple import SimpleLSTM
from src.data.dataset import get_dataloaders

SEP = "=" * 60

def run_analysis():
    # ─────────────────────────────────────────
    # 1. CHECKPOINT ANALIZI
    # ─────────────────────────────────────────
    print("\n" + SEP)
    print("1. CHECKPOINT ANALIZI")
    print(SEP)

    ckpt_path = MODEL_DIR / "best_model.pth"
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    epoch    = ckpt.get('epoch', '?')
    best_val = ckpt.get('best_val_acc', 0)
    history  = ckpt.get('history', {})

    train_acc  = history.get('train_acc', [])
    val_acc    = history.get('val_acc', [])
    train_loss = history.get('train_loss', [])
    val_loss   = history.get('val_loss', [])
    lrs        = history.get('lr', [])

    print(f"Son epoch         : {epoch}")
    print(f"Best Val Acc      : {best_val:.4f}%")
    print(f"Egitilen epoch    : {len(train_acc)}")

    gap_per_epoch = []
    if train_acc and val_acc:
        gap_per_epoch = [t - v for t, v in zip(train_acc, val_acc)]
        max_gap_ep  = gap_per_epoch.index(max(gap_per_epoch)) + 1
        best_val_ep = val_acc.index(max(val_acc)) + 1

        print(f"\n{'Epoch':>5} | {'TrainAcc':>9} | {'ValAcc':>8} | {'Gap':>8} | {'TrainLoss':>10} | {'ValLoss':>8}")
        print("-" * 65)
        for i in range(len(train_acc)):
            gap  = gap_per_epoch[i]
            note = ""
            if (i + 1) == best_val_ep: note = " <- BEST VAL"
            if (i + 1) == max_gap_ep:  note += " <- MAX GAP"
            print(f"{i+1:>5} | {train_acc[i]:>8.2f}% | {val_acc[i]:>7.2f}% | {gap:>7.2f}% | {train_loss[i]:>10.4f} | {val_loss[i]:>8.4f}{note}")

        print(f"\nEn iyi Val Acc : {max(val_acc):.2f}%  (epoch {best_val_ep})")
        print(f"Son Gap        : {gap_per_epoch[-1]:.2f}%  (train - val)")
        print(f"Max Gap        : {max(gap_per_epoch):.2f}%  (epoch {max_gap_ep})")
        if max(gap_per_epoch) > 20:
            print(">>> OVERFITTING TESPIT EDILDI! Gap > 20%")

    # ─────────────────────────────────────────
    # 2. MODEL MIMARISI
    # ─────────────────────────────────────────
    print("\n" + SEP)
    print("2. MODEL MIMARISI")
    print(SEP)

    model = SimpleLSTM(input_size=LANDMARK_FEATURES, num_classes=NUM_CLASSES)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre  : {total_params:,}")
    print(f"Model boyutu (MB) : {total_params * 4 / 1024**2:.1f} MB")
    print("\nKatman bazli parametre sayisi:")
    for name, param in model.named_parameters():
        print(f"  {name:<40} {param.numel():>10,}  shape={list(param.shape)}")

    # ─────────────────────────────────────────
    # 3. SINIF BAZLI DOGRULUK
    # ─────────────────────────────────────────
    print("\n" + SEP)
    print("3. SINIF BAZLI DOGRULUK (validation seti)")
    print(SEP)

    device = torch.device(DEVICE)
    model  = model.to(device)

    all_preds = all_labels = all_confs = None
    class_accs = {}

    try:
        # num_workers=0 — Windows multiprocessing fix
        _, val_loader, _ = get_dataloaders(mode="landmarks", batch_size=64, num_workers=0)

        class_correct = defaultdict(int)
        class_total   = defaultdict(int)
        preds_list    = []
        labels_list   = []
        confs_list    = []

        print("Val loader uzerinde inference yapiliyor...")
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                lm     = batch[0]
                labels = batch[-1]
                lm, labels = lm.to(device), labels.to(device)

                logits = model(lm)
                probs  = torch.softmax(logits, dim=1)
                preds  = probs.argmax(dim=1)
                confs  = probs.max(dim=1).values

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                confs_list.extend(confs.cpu().numpy())

                for pred, label in zip(preds, labels):
                    class_total[label.item()] += 1
                    if pred == label:
                        class_correct[label.item()] += 1

                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch {batch_idx+1} islendi...")

        all_preds  = np.array(preds_list)
        all_labels = np.array(labels_list)
        all_confs  = np.array(confs_list)

        overall_acc  = (all_preds == all_labels).mean() * 100
        correct_mask = (all_preds == all_labels)

        print(f"\nOverall Val Accuracy     : {overall_acc:.2f}%")
        print(f"Toplam ornek             : {len(all_labels)}")
        print(f"Mean Confidence          : {all_confs.mean()*100:.1f}%")
        print(f"Dogru tahminde conf ort  : {all_confs[correct_mask].mean()*100:.1f}%")
        print(f"Yanlis tahminde conf ort : {all_confs[~correct_mask].mean()*100:.1f}%")

        for cls in sorted(class_total.keys()):
            if class_total[cls] > 0:
                class_accs[cls] = class_correct[cls] / class_total[cls] * 100

        sorted_by_acc = sorted(class_accs.items(), key=lambda x: x[1])

        print(f"\n--- En Kotü 15 Sinif ---")
        for cls, acc in sorted_by_acc[:15]:
            print(f"  ClassID={cls:3d}  Acc={acc:5.1f}%  ({class_correct[cls]}/{class_total[cls]})")

        print(f"\n--- En Iyi 15 Sinif ---")
        for cls, acc in sorted_by_acc[-15:][::-1]:
            print(f"  ClassID={cls:3d}  Acc={acc:5.1f}%  ({class_correct[cls]}/{class_total[cls]})")

        zero_cls    = [cls for cls, acc in class_accs.items() if acc == 0.0]
        perfect_cls = [cls for cls, acc in class_accs.items() if acc == 100.0]
        print(f"\n%0   dogruluklu sinif  : {len(zero_cls)}")
        print(f"%100 dogruluklu sinif  : {len(perfect_cls)}")
        if zero_cls:
            print(f"  Sifir siniflar: {zero_cls[:30]}")

    except Exception as e:
        print(f"[HATA] Validation loader: {e}")
        import traceback; traceback.print_exc()

    # ─────────────────────────────────────────
    # 4. PIPELINE TUTARSIZLIK RAPORU
    # ─────────────────────────────────────────
    print("\n" + SEP)
    print("4. PIPELINE TUTARSIZLIK RAPORU")
    print(SEP)

    issues = [
        ("[KRITIK ] realtime.py",
         "VIDEO mode kullaniyor, egitim IMAGE mode ile yapildi",
         "Landmark degerleri farkli -> dogruluk duser",
         "IMAGE mode + detect() kullan (pytorch_predictor.py'deki gibi)"),
        ("[ONEMLI ] realtime.py _predict_sign()",
         "Frame padding: repeat-last; egitimde lineer interpolasyon",
         "Sequence input kalitesi dusuyor",
         "pytorch_predictor.py'deki lineer interpolasyonu kopyala"),
        ("[ORTA   ] realtime.py vs pytorch_predictor.py",
         "Motion esik degerleri farkli: MOTION 0.010 vs 0.008",
         "Ayni harekete farkli tepki, tutarsiz segmentation",
         "config.py'de ortak sabitlere al"),
        ("[ORTA   ] pytorch_predictor.py:38",
         "SimpleLSTM() parametresiz cagriliyor (default'lardan aliyor)",
         "Config degisirse boyut uyumsuzlugu riski",
         "SimpleLSTM(input_size=LANDMARK_FEATURES, num_classes=NUM_CLASSES) yaz"),
    ]

    for title, sorun, etki, cozum in issues:
        print(f"\n{title}")
        print(f"  Sorun : {sorun}")
        print(f"  Etki  : {etki}")
        print(f"  Cozum : {cozum}")

    # ─────────────────────────────────────────
    # 5. GRAFIK CIKAR
    # ─────────────────────────────────────────
    if not (train_acc and val_acc):
        print("\n[UYARI] History bulunamadi, grafik atlaniyor.")
        return

    print("\n" + SEP)
    print("5. GRAFIK OLUSTURULUYOR")
    print(SEP)

    epochs = list(range(1, len(train_acc) + 1))
    gap    = gap_per_epoch

    fig = plt.figure(figsize=(20, 14), facecolor='#0f0f1a')
    gs  = gridspec.GridSpec(2, 3, hspace=0.42, wspace=0.35)

    BG   = '#1a1a2e'; GRID = '#2a2a3a'; TXT  = '#e0e0e0'
    C_TR = '#4fc3f7'; C_VL = '#ff8a65'; C_GP = '#ce93d8'
    C_LR = '#a5d6a7'; C_ER = '#ef5350'

    def style(ax, title):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TXT, labelsize=9)
        ax.set_title(title, color=TXT, fontsize=11, pad=8, fontweight='bold')
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
        for sp in ax.spines.values(): sp.set_edgecolor(GRID)
        ax.xaxis.label.set_color(TXT); ax.yaxis.label.set_color(TXT)

    # Panel 1: Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(epochs, train_loss, color=C_TR, label='Train', linewidth=2)
    ax1.plot(epochs, val_loss,   color=C_VL, label='Val',   linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.legend(facecolor=GRID, labelcolor=TXT, fontsize=9)
    style(ax1, 'Loss Curve')

    # Panel 2: Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, train_acc, color=C_TR, label='Train Acc', linewidth=2)
    ax2.plot(epochs, val_acc,   color=C_VL, label='Val Acc',   linewidth=2)
    bep = val_acc.index(max(val_acc)) + 1
    ax2.axvline(bep, color='#ffeb3b', linestyle='--', alpha=0.7, lw=1.5, label=f'Best ep{bep}')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy (%)')
    ax2.legend(facecolor=GRID, labelcolor=TXT, fontsize=9)
    style(ax2, 'Accuracy Curve')

    # Panel 3: Overfitting Gap
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.fill_between(epochs, gap, alpha=0.35, color=C_GP)
    ax3.plot(epochs, gap, color=C_GP, linewidth=2)
    ax3.axhline(0,  color='white', linestyle='-',  alpha=0.25, lw=1)
    ax3.axhline(20, color=C_ER,   linestyle='--', alpha=0.6,  lw=1.5, label='20% danger')
    ax3.set_xlabel('Epoch'); ax3.set_ylabel('Gap (%)')
    ax3.legend(facecolor=GRID, labelcolor=TXT, fontsize=9)
    style(ax3, 'Overfitting Gap  (Train - Val)')

    # Panel 4: LR
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.semilogy(epochs, lrs, color=C_LR, linewidth=2)
    ax4.set_xlabel('Epoch'); ax4.set_ylabel('LR (log scale)')
    style(ax4, 'Learning Rate Schedule')

    # Panel 5: Confidence Dist
    ax5 = fig.add_subplot(gs[1, 1])
    if all_confs is not None:
        bins = np.arange(0, 1.05, 0.05)
        cm   = (all_preds == all_labels)
        ax5.hist(all_confs[cm],  bins=bins, alpha=0.7, color=C_VL, label='Correct')
        ax5.hist(all_confs[~cm], bins=bins, alpha=0.7, color=C_ER, label='Wrong')
        ax5.set_xlabel('Confidence'); ax5.set_ylabel('Count')
        ax5.legend(facecolor=GRID, labelcolor=TXT, fontsize=9)
    else:
        ax5.text(0.5, 0.5, 'Val loader unavailable', ha='center', va='center',
                 color=TXT, transform=ax5.transAxes)
    style(ax5, 'Confidence Distribution (Val Set)')

    # Panel 6: Ozet
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor(BG); ax6.axis('off')
    for sp in ax6.spines.values(): sp.set_edgecolor(GRID)

    overall_str = f"{overall_acc:.2f}%" if all_confs is not None else "N/A"
    lines = [
        f"Best Val Acc: {max(val_acc):.2f}% (ep {bep})",
        f"Runtime Val:  {overall_str}",
        f"Final Train:  {train_acc[-1]:.2f}%",
        f"Final Val:    {val_acc[-1]:.2f}%",
        f"Final Gap:    {gap[-1]:.2f}%",
        f"Max Gap:      {max(gap):.2f}% (ep {gap.index(max(gap))+1})",
        f"Zero classes: {len([c for c,a in class_accs.items() if a==0])}",
        f"100% classes: {len([c for c,a in class_accs.items() if a==100])}",
        "",
        "-----  KRITIK SORUNLAR  -----",
        " realtime.py: VIDEO mode",
        "  (egitim IMAGE mode ile yapildi)",
        " Frame interpolasyon tutarsiz",
        "",
        "-----  ORTA SORUNLAR  -----",
        " Motion threshold farki",
        " SimpleLSTM() parametresiz",
        "",
        "-----  YENI MODEL ONERISI  -----",
        " C2fLSTM (YOLOv8-inspired)",
        " Batch=16 | EarlyStop=15",
        " LabelSmooth=0.1 | Mixup",
    ]

    for j, line in enumerate(lines):
        c = TXT
        if 'KRITIK' in line: c = C_ER
        if 'ORTA' in line:   c = '#ffb74d'
        if 'YENI' in line or 'ONERISI' in line: c = C_LR
        if 'Best' in line:   c = C_VL
        ax6.text(0.04, 0.97 - j * 0.048, line, transform=ax6.transAxes,
                 color=c, fontsize=8.5, va='top', fontfamily='monospace')
    style(ax6, 'Analysis Summary')

    fig.suptitle('DeepSign-TID  --  Mevcut Model Analizi (SimpleLSTM)',
                 color=TXT, fontsize=14, fontweight='bold', y=0.98)

    out = Path('analysis_current_model.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0f0f1a')
    print(f"Grafik kaydedildi: {out.resolve()}")

    print("\n" + SEP)
    print("ANALIZ TAMAMLANDI")
    print(SEP)


if __name__ == '__main__':
    run_analysis()
