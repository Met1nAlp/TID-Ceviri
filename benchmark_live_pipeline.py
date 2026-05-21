"""
Benchmark the live sign-recognition pipeline on AUTSL videos.

This script replays real split videos frame-by-frame through the same
segmentation + prediction logic used by the web app, then reports:
  - prediction coverage
  - top-1 / top-3 accuracy
  - early prediction rate
  - per-class failures

Example:
    python benchmark_live_pipeline.py --split test --sample-per-class 2
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from app.pytorch_predictor import PyTorchPredictor


BASE_DIR = Path(__file__).resolve().parent
AUTSL_DIR = BASE_DIR / "AUTSL"
LOG_DIR = BASE_DIR / "logs" / "benchmarks"
LOG_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class VideoSample:
    video_name: str
    label_id: int
    video_path: Path


def load_split_samples(split: str) -> list[VideoSample]:
    split_csv = AUTSL_DIR / f"{split}.csv"
    video_dir = AUTSL_DIR / split
    samples: list[VideoSample] = []

    with split_csv.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 2:
                continue
            video_name, label_text = row[0].strip(), row[1].strip()
            video_path = video_dir / video_name
            if not video_path.exists():
                continue
            samples.append(
                VideoSample(
                    video_name=video_name,
                    label_id=int(label_text),
                    video_path=video_path,
                )
            )

    return samples


def select_samples(
    samples: list[VideoSample],
    sample_per_class: int | None,
    limit: int | None,
    seed: int,
) -> list[VideoSample]:
    if sample_per_class is None and limit is None:
        return samples

    rng = random.Random(seed)
    by_class: dict[int, list[VideoSample]] = defaultdict(list)
    for sample in samples:
        by_class[sample.label_id].append(sample)

    selected: list[VideoSample] = []
    class_ids = sorted(by_class.keys())
    rng.shuffle(class_ids)

    if sample_per_class is not None:
        for class_id in class_ids:
            class_samples = by_class[class_id][:]
            rng.shuffle(class_samples)
            selected.extend(class_samples[:sample_per_class])
    else:
        selected = samples[:]

    rng.shuffle(selected)

    if limit is not None:
        selected = selected[:limit]

    return selected


def create_predictor(device: str) -> PyTorchPredictor:
    return PyTorchPredictor(
        device=device,
        enable_temporal_smoothing=True,
        use_video_landmarkers=True,
        swap_handedness=False,
        motion_threshold=0.0080,
        idle_threshold=0.0060,
        min_sign_frames=15,
        idle_frames_to_stop=12,
        start_frames=2,
    )


def replay_video(
    predictor: PyTorchPredictor,
    sample: VideoSample,
) -> dict:
    predictor.reset_stream_state()
    capture = cv2.VideoCapture(str(sample.video_path))

    if not capture.isOpened():
        return {
            "video_name": sample.video_name,
            "label_id": sample.label_id,
            "status": "video_open_failed",
        }

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    frame_index = 0
    first_prediction = None
    last_frame = None

    while True:
        success, frame = capture.read()
        if not success:
            break

        frame_index += 1
        last_frame = frame
        predictions, _, _ = predictor.process_frame(frame)
        if predictions and first_prediction is None:
            first_prediction = {
                "phase": "stream",
                "frame_index": frame_index,
                "predictions": predictions,
            }

    capture.release()

    flush_frames = 0
    if first_prediction is None and last_frame is not None:
        for _ in range(predictor.IDLE_FRAMES_TO_STOP + 2):
            flush_frames += 1
            predictions, _, _ = predictor.process_frame(last_frame)
            if predictions:
                first_prediction = {
                    "phase": "flush",
                    "frame_index": frame_index + flush_frames,
                    "predictions": predictions,
                }
                break

    result = {
        "video_name": sample.video_name,
        "label_id": sample.label_id,
        "status": "ok",
        "frames_total": frame_index if total_frames == 0 else total_frames,
        "flush_frames": flush_frames,
    }

    if first_prediction is None:
        result.update(
            {
                "prediction_found": False,
                "prediction_phase": "none",
                "prediction_frame_index": None,
                "progress_ratio": None,
                "predicted_class_id": None,
                "predicted_label_tr": None,
                "predicted_confidence": None,
                "top3_class_ids": [],
                "top3_labels_tr": [],
                "top1_correct": False,
                "top3_correct": False,
            }
        )
        return result

    top_predictions = first_prediction["predictions"]
    top_prediction = top_predictions[0]
    progress_denominator = max(1, frame_index)
    progress_ratio = min(1.0, first_prediction["frame_index"] / progress_denominator)

    result.update(
        {
            "prediction_found": True,
            "prediction_phase": first_prediction["phase"],
            "prediction_frame_index": first_prediction["frame_index"],
            "progress_ratio": round(progress_ratio, 4),
            "predicted_class_id": int(top_prediction["class_id"]),
            "predicted_label_tr": top_prediction["label_tr"],
            "predicted_confidence": float(top_prediction["confidence"]),
            "top3_class_ids": [int(item["class_id"]) for item in top_predictions],
            "top3_labels_tr": [item["label_tr"] for item in top_predictions],
            "top1_correct": int(top_prediction["class_id"]) == sample.label_id,
            "top3_correct": sample.label_id in [int(item["class_id"]) for item in top_predictions],
        }
    )
    return result


def summarise_results(results: list[dict], class_labels: dict[int, tuple[str, str]]) -> dict:
    ok_results = [item for item in results if item.get("status") == "ok"]
    predicted_results = [item for item in ok_results if item.get("prediction_found")]
    total = len(ok_results)
    predicted_total = len(predicted_results)

    top1_all = sum(1 for item in ok_results if item.get("top1_correct")) / total * 100 if total else 0.0
    top3_all = sum(1 for item in ok_results if item.get("top3_correct")) / total * 100 if total else 0.0
    coverage = predicted_total / total * 100 if total else 0.0

    top1_predicted = (
        sum(1 for item in predicted_results if item.get("top1_correct")) / predicted_total * 100
        if predicted_total
        else 0.0
    )
    top3_predicted = (
        sum(1 for item in predicted_results if item.get("top3_correct")) / predicted_total * 100
        if predicted_total
        else 0.0
    )

    stream_predictions = [
        item for item in predicted_results
        if item.get("prediction_phase") == "stream" and item.get("progress_ratio") is not None
    ]
    early_predictions = [
        item for item in stream_predictions
        if float(item["progress_ratio"]) < 0.85
    ]

    per_class = defaultdict(lambda: {"total": 0, "top1": 0, "top3": 0})
    confusion = Counter()
    for item in ok_results:
        class_id = int(item["label_id"])
        per_class[class_id]["total"] += 1
        if item.get("top1_correct"):
            per_class[class_id]["top1"] += 1
        if item.get("top3_correct"):
            per_class[class_id]["top3"] += 1
        if item.get("prediction_found") and not item.get("top1_correct"):
            confusion[(class_id, int(item["predicted_class_id"]))] += 1

    class_rows = []
    for class_id, stats in per_class.items():
        top1_acc = stats["top1"] / stats["total"] * 100 if stats["total"] else 0.0
        class_rows.append(
            {
                "class_id": class_id,
                "label_tr": class_labels.get(class_id, (f"class_{class_id}", ""))[0],
                "samples": stats["total"],
                "top1_correct": stats["top1"],
                "top3_correct": stats["top3"],
                "top1_acc": round(top1_acc, 2),
                "top3_acc": round((stats["top3"] / stats["total"] * 100) if stats["total"] else 0.0, 2),
            }
        )
    class_rows.sort(key=lambda item: (item["top1_acc"], item["class_id"]))

    common_confusions = []
    for (true_id, pred_id), count in confusion.most_common(15):
        common_confusions.append(
            {
                "true_class_id": true_id,
                "true_label_tr": class_labels.get(true_id, (f"class_{true_id}", ""))[0],
                "pred_class_id": pred_id,
                "pred_label_tr": class_labels.get(pred_id, (f"class_{pred_id}", ""))[0],
                "count": count,
            }
        )

    summary = {
        "total_samples": total,
        "prediction_coverage": round(coverage, 2),
        "top1_accuracy_all": round(top1_all, 2),
        "top3_accuracy_all": round(top3_all, 2),
        "top1_accuracy_predicted_only": round(top1_predicted, 2),
        "top3_accuracy_predicted_only": round(top3_predicted, 2),
        "stream_prediction_count": len(stream_predictions),
        "early_prediction_count": len(early_predictions),
        "early_prediction_rate": round(
            (len(early_predictions) / len(stream_predictions) * 100) if stream_predictions else 0.0,
            2,
        ),
        "median_prediction_progress": round(
            statistics.median(item["progress_ratio"] for item in stream_predictions),
            4,
        ) if stream_predictions else None,
        "class_rows": class_rows,
        "hardest_classes": class_rows[:20],
        "common_confusions": common_confusions,
    }
    return summary


def save_report(results: list[dict], summary: dict, split: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = LOG_DIR / f"live_pipeline_{split}_{timestamp}.json"
    payload = {
        "created_at": timestamp,
        "split": split,
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def save_video_csv(results: list[dict], class_labels: dict[int, tuple[str, str]], split: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = LOG_DIR / f"live_pipeline_{split}_{timestamp}_videos.csv"
    fieldnames = [
        "video_name",
        "label_id",
        "label_tr",
        "prediction_found",
        "prediction_phase",
        "prediction_frame_index",
        "progress_ratio",
        "predicted_class_id",
        "predicted_label_tr",
        "predicted_confidence",
        "top1_correct",
        "top3_correct",
        "top3_labels_tr",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            label_id = int(item["label_id"])
            writer.writerow(
                {
                    "video_name": item["video_name"],
                    "label_id": label_id,
                    "label_tr": class_labels.get(label_id, (f"class_{label_id}", ""))[0],
                    "prediction_found": item.get("prediction_found", False),
                    "prediction_phase": item.get("prediction_phase"),
                    "prediction_frame_index": item.get("prediction_frame_index"),
                    "progress_ratio": item.get("progress_ratio"),
                    "predicted_class_id": item.get("predicted_class_id"),
                    "predicted_label_tr": item.get("predicted_label_tr"),
                    "predicted_confidence": item.get("predicted_confidence"),
                    "top1_correct": item.get("top1_correct", False),
                    "top3_correct": item.get("top3_correct", False),
                    "top3_labels_tr": " | ".join(item.get("top3_labels_tr", [])),
                }
            )

    return output_path


def save_class_csv(summary: dict, split: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = LOG_DIR / f"live_pipeline_{split}_{timestamp}_classes.csv"
    fieldnames = [
        "class_id",
        "label_tr",
        "samples",
        "top1_correct",
        "top3_correct",
        "top1_acc",
        "top3_acc",
    ]

    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary["class_rows"]:
            writer.writerow(row)

    return output_path


def print_class_summary(summary: dict, limit: int = 15):
    class_rows = summary["class_rows"]
    best_rows = sorted(class_rows, key=lambda row: (-row["top1_acc"], row["class_id"]))[:limit]
    worst_rows = sorted(class_rows, key=lambda row: (row["top1_acc"], row["class_id"]))[:limit]

    print("\nEN IYI KELIMELER")
    print("-" * 70)
    for row in best_rows:
        print(
            f"{row['label_tr']:<22} "
            f"top1=%{row['top1_acc']:>6.2f} "
            f"top3=%{row['top3_acc']:>6.2f} "
            f"n={row['samples']}"
        )

    print("\nEN KOTU KELIMELER")
    print("-" * 70)
    for row in worst_rows:
        print(
            f"{row['label_tr']:<22} "
            f"top1=%{row['top1_acc']:>6.2f} "
            f"top3=%{row['top3_acc']:>6.2f} "
            f"n={row['samples']}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark the live TID pipeline on AUTSL videos.")
    parser.add_argument("--split", choices=["val", "test", "train"], default="test")
    parser.add_argument("--sample-per-class", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    samples = load_split_samples(args.split)
    selected_samples = select_samples(
        samples=samples,
        sample_per_class=args.sample_per_class,
        limit=args.limit,
        seed=args.seed,
    )

    predictor = create_predictor(device=args.device)

    print("=" * 70)
    print("LIVE PIPELINE BENCHMARK")
    print("=" * 70)
    print(f"Split: {args.split}")
    print(f"Total split samples: {len(samples)}")
    print(f"Selected samples: {len(selected_samples)}")
    print(f"Device: {predictor.device}")
    print("=" * 70)

    results = []
    for index, sample in enumerate(selected_samples, start=1):
        result = replay_video(predictor, sample)
        results.append(result)

        if index % 10 == 0 or index == len(selected_samples):
            correct = sum(1 for item in results if item.get("top1_correct"))
            predicted = sum(1 for item in results if item.get("prediction_found"))
            print(
                f"[{index}/{len(selected_samples)}] "
                f"coverage={predicted/max(1, len(results))*100:.1f}% "
                f"top1={correct/max(1, len(results))*100:.1f}%"
            )

    summary = summarise_results(results, predictor.class_labels)
    report_path = save_report(results, summary, args.split)
    video_csv_path = save_video_csv(results, predictor.class_labels, args.split)
    class_csv_path = save_class_csv(summary, args.split)

    print("\nSUMMARY")
    print("-" * 70)
    print(f"Coverage: {summary['prediction_coverage']:.2f}%")
    print(f"Top-1 (all): {summary['top1_accuracy_all']:.2f}%")
    print(f"Top-3 (all): {summary['top3_accuracy_all']:.2f}%")
    print(f"Top-1 (predicted only): {summary['top1_accuracy_predicted_only']:.2f}%")
    print(f"Early prediction rate: {summary['early_prediction_rate']:.2f}%")
    print(f"Median prediction progress: {summary['median_prediction_progress']}")
    print_class_summary(summary)
    print("\nDOSYALAR")
    print("-" * 70)
    print(f"JSON rapor  : {report_path}")
    print(f"Video CSV   : {video_csv_path}")
    print(f"Sinif CSV   : {class_csv_path}")


if __name__ == "__main__":
    main()
