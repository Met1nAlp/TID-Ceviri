"""
Analyze live benchmark reports and suggest targeted next steps.

This script reads a JSON report produced by benchmark_live_pipeline.py and
generates:
  - per-class action recommendations
  - dominant confusion pairs
  - a short priority list for targeted fine-tuning

Example:
    python analyze_benchmark_report.py --report logs/benchmarks/live_pipeline_test_20260509_135842.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = BASE_DIR / "logs" / "benchmarks"


def find_latest_report() -> Path:
    reports = sorted(BENCHMARK_DIR.glob("live_pipeline_*.json"))
    if not reports:
        raise FileNotFoundError("No benchmark report found under logs/benchmarks.")
    return reports[-1]


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_class_lookup(report: dict) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for row in report.get("summary", {}).get("class_rows", []):
        lookup[int(row["class_id"])] = row["label_tr"]

    for item in report.get("results", []):
        class_id = int(item["label_id"])
        lookup.setdefault(class_id, item.get("label_tr", f"class_{class_id}"))
        predicted_class_id = item.get("predicted_class_id")
        predicted_label = item.get("predicted_label_tr")
        if predicted_class_id is not None and predicted_label:
            lookup.setdefault(int(predicted_class_id), predicted_label)

    return lookup


def analyse_report(report: dict) -> dict:
    class_lookup = build_class_lookup(report)
    per_class: dict[int, dict] = {}
    pair_stats: dict[tuple[int, int], dict] = defaultdict(
        lambda: {
            "count": 0,
            "top3_rescued": 0,
            "confidences": [],
        }
    )

    for item in report.get("results", []):
        if item.get("status") != "ok":
            continue

        true_id = int(item["label_id"])
        stats = per_class.setdefault(
            true_id,
            {
                "class_id": true_id,
                "label_tr": class_lookup.get(true_id, f"class_{true_id}"),
                "samples": 0,
                "top1_correct": 0,
                "top3_correct": 0,
                "no_prediction": 0,
                "top3_hit_not_top1": 0,
                "top3_miss": 0,
                "prediction_progress": [],
                "wrong_top1_counter": Counter(),
                "top3_partner_counter": Counter(),
            },
        )

        stats["samples"] += 1
        if item.get("prediction_found"):
            progress_ratio = item.get("progress_ratio")
            if progress_ratio is not None:
                stats["prediction_progress"].append(float(progress_ratio))
        else:
            stats["no_prediction"] += 1
            continue

        if item.get("top1_correct"):
            stats["top1_correct"] += 1
        if item.get("top3_correct"):
            stats["top3_correct"] += 1

        if item.get("top3_correct") and not item.get("top1_correct"):
            stats["top3_hit_not_top1"] += 1
            for class_id in item.get("top3_class_ids", []):
                class_id = int(class_id)
                if class_id != true_id:
                    stats["top3_partner_counter"][class_id] += 1

        if not item.get("top3_correct"):
            stats["top3_miss"] += 1

        predicted_id = item.get("predicted_class_id")
        if predicted_id is not None and not item.get("top1_correct"):
            predicted_id = int(predicted_id)
            stats["wrong_top1_counter"][predicted_id] += 1
            pair = pair_stats[(true_id, predicted_id)]
            pair["count"] += 1
            if item.get("top3_correct"):
                pair["top3_rescued"] += 1
            confidence = item.get("predicted_confidence")
            if confidence is not None:
                pair["confidences"].append(float(confidence))

    class_rows = []
    for class_id, stats in per_class.items():
        samples = max(1, stats["samples"])
        top1_acc = stats["top1_correct"] / samples * 100.0
        top3_acc = stats["top3_correct"] / samples * 100.0
        rescue_gap = top3_acc - top1_acc
        dominant_top1 = stats["wrong_top1_counter"].most_common(1)
        dominant_top3 = stats["top3_partner_counter"].most_common(1)

        dominant_pred_id = dominant_top1[0][0] if dominant_top1 else None
        dominant_pred_count = dominant_top1[0][1] if dominant_top1 else 0
        dominant_top3_partner_id = dominant_top3[0][0] if dominant_top3 else None

        if top3_acc >= 50.0 and rescue_gap >= 25.0:
            action = "top3_salvageable"
            rationale = "Dogru kelime siklikla ilk 3'e giriyor, siralama/ayirt etme sorunu var."
        elif top3_acc < 40.0:
            action = "targeted_finetune"
            rationale = "Dogru kelime cogu denemede ilk 3'e bile girmiyor."
        else:
            action = "mixed_review"
            rationale = "Kismi dogruluk var; hem veri hem segmentasyon etkisi olabilir."

        if stats["no_prediction"] > 0:
            rationale += " Bazi denemelerde hic tahmin uretilmiyor."

        priority_score = round(
            (100.0 - top1_acc) * samples
            + stats["top3_miss"] * 18
            + stats["top3_hit_not_top1"] * 10
            + stats["no_prediction"] * 12,
            2,
        )

        class_rows.append(
            {
                "class_id": class_id,
                "label_tr": stats["label_tr"],
                "samples": stats["samples"],
                "top1_correct": stats["top1_correct"],
                "top3_correct": stats["top3_correct"],
                "top1_acc": round(top1_acc, 2),
                "top3_acc": round(top3_acc, 2),
                "rescue_gap": round(rescue_gap, 2),
                "no_prediction": stats["no_prediction"],
                "top3_hit_not_top1": stats["top3_hit_not_top1"],
                "top3_miss": stats["top3_miss"],
                "dominant_confusion_class_id": dominant_pred_id if dominant_pred_id is not None else -1,
                "dominant_confusion_label": class_lookup.get(dominant_pred_id, "-")
                if dominant_pred_id is not None
                else "-",
                "dominant_confusion_count": dominant_pred_count,
                "top3_partner_class_id": dominant_top3_partner_id
                if dominant_top3_partner_id is not None
                else -1,
                "top3_partner_label": class_lookup.get(dominant_top3_partner_id, "-")
                if dominant_top3_partner_id is not None
                else "-",
                "recommended_action": action,
                "rationale": rationale,
                "priority_score": priority_score,
            }
        )

    class_rows.sort(
        key=lambda row: (
            {"targeted_finetune": 0, "top3_salvageable": 1, "mixed_review": 2}.get(
                row["recommended_action"], 9
            ),
            -row["priority_score"],
            row["class_id"],
        )
    )

    pair_rows = []
    for (true_id, pred_id), stats in sorted(
        pair_stats.items(),
        key=lambda item: (-item[1]["count"], item[0][0], item[0][1]),
    ):
        rescue_rate = (
            stats["top3_rescued"] / stats["count"] * 100.0 if stats["count"] else 0.0
        )
        pair_rows.append(
            {
                "true_class_id": true_id,
                "true_label_tr": class_lookup.get(true_id, f"class_{true_id}"),
                "pred_class_id": pred_id,
                "pred_label_tr": class_lookup.get(pred_id, f"class_{pred_id}"),
                "count": stats["count"],
                "top3_rescued": stats["top3_rescued"],
                "rescue_rate": round(rescue_rate, 2),
                "mean_wrong_confidence": round(
                    sum(stats["confidences"]) / len(stats["confidences"]), 2
                )
                if stats["confidences"]
                else 0.0,
            }
        )

    summary = {
        "total_classes": len(class_rows),
        "top3_salvageable_classes": sum(
            1 for row in class_rows if row["recommended_action"] == "top3_salvageable"
        ),
        "targeted_finetune_classes": sum(
            1 for row in class_rows if row["recommended_action"] == "targeted_finetune"
        ),
        "mixed_review_classes": sum(
            1 for row in class_rows if row["recommended_action"] == "mixed_review"
        ),
        "top_priority_classes": class_rows[:20],
        "top_confusion_pairs": pair_rows[:30],
    }

    return {
        "source_created_at": report.get("created_at"),
        "source_split": report.get("split"),
        "summary": summary,
        "class_rows": class_rows,
        "pair_rows": pair_rows,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(analysis: dict, source_report: Path) -> tuple[Path, Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = source_report.stem
    json_path = BENCHMARK_DIR / f"{stem}_analysis_{timestamp}.json"
    class_csv_path = BENCHMARK_DIR / f"{stem}_analysis_classes_{timestamp}.csv"
    pair_csv_path = BENCHMARK_DIR / f"{stem}_analysis_pairs_{timestamp}.csv"

    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(class_csv_path, analysis["class_rows"])
    write_csv(pair_csv_path, analysis["pair_rows"])
    return json_path, class_csv_path, pair_csv_path


def print_summary(analysis: dict) -> None:
    summary = analysis["summary"]
    print("=" * 70)
    print("BENCHMARK ANALIZI")
    print("=" * 70)
    print(f"Top-3 ile kurtarilabilir sinif: {summary['top3_salvageable_classes']}")
    print(f"Hedefli fine-tune gereken sinif: {summary['targeted_finetune_classes']}")
    print(f"Karisik inceleme gereken sinif: {summary['mixed_review_classes']}")

    print("\nONCELIKLI SINIFLAR")
    print("-" * 70)
    for row in summary["top_priority_classes"][:12]:
        print(
            f"{row['label_tr']:<22} "
            f"top1=%{row['top1_acc']:>6.2f} "
            f"top3=%{row['top3_acc']:>6.2f} "
            f"aksiyon={row['recommended_action']:<18} "
            f"karisan={row['dominant_confusion_label']}"
        )

    print("\nEN SIK KARISAN CIFTLER")
    print("-" * 70)
    for row in summary["top_confusion_pairs"][:12]:
        print(
            f"{row['true_label_tr']:<20} -> {row['pred_label_tr']:<20} "
            f"n={row['count']:<3} top3_kurtarma=%{row['rescue_rate']:>6.2f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a live benchmark report.")
    parser.add_argument("--report", type=Path, default=None, help="Path to benchmark JSON report.")
    args = parser.parse_args()

    report_path = args.report if args.report is not None else find_latest_report()
    report_path = report_path.resolve()

    report = load_report(report_path)
    analysis = analyse_report(report)
    json_path, class_csv_path, pair_csv_path = save_outputs(analysis, report_path)

    print_summary(analysis)
    print("\nDOSYALAR")
    print("-" * 70)
    print(f"Kaynak rapor : {report_path}")
    print(f"Analiz JSON  : {json_path}")
    print(f"Sinif CSV    : {class_csv_path}")
    print(f"Pair CSV     : {pair_csv_path}")


if __name__ == "__main__":
    main()
