"""
Helpers for targeted fine-tuning from benchmark analysis outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def load_focus_bundle(
    analysis_report_path: str | Path,
    actions: list[str] | tuple[str, ...],
    top_n: int | None = None,
    include_confusion_partners: bool = True,
) -> dict:
    report_path = Path(analysis_report_path)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    class_rows = payload.get("class_rows", [])

    selected_rows = [
        row for row in class_rows
        if row.get("recommended_action") in set(actions)
    ]
    selected_rows.sort(
        key=lambda row: (-float(row.get("priority_score", 0.0)), int(row.get("class_id", -1)))
    )
    if top_n is not None and top_n > 0:
        selected_rows = selected_rows[:top_n]

    focus_class_ids = sorted({int(row["class_id"]) for row in selected_rows})
    partner_class_ids = set()
    if include_confusion_partners:
        for row in selected_rows:
            for key in ("dominant_confusion_class_id", "top3_partner_class_id"):
                class_id = int(row.get(key, -1))
                if class_id >= 0 and class_id not in focus_class_ids:
                    partner_class_ids.add(class_id)

    return {
        "report_path": str(report_path),
        "actions": list(actions),
        "focus_class_ids": focus_class_ids,
        "partner_class_ids": sorted(partner_class_ids),
        "selected_rows": selected_rows,
    }


def build_weighted_training_config(
    num_classes: int,
    focus_class_ids: list[int],
    partner_class_ids: list[int] | None = None,
    focus_sample_boost: float = 3.0,
    partner_sample_boost: float = 1.75,
    focus_loss_boost: float = 2.0,
    partner_loss_boost: float = 1.35,
    device: str = "cpu",
) -> dict:
    partner_class_ids = partner_class_ids or []

    sample_weight_map = {class_id: float(focus_sample_boost) for class_id in focus_class_ids}
    for class_id in partner_class_ids:
        sample_weight_map[class_id] = max(
            sample_weight_map.get(class_id, 1.0),
            float(partner_sample_boost),
        )

    loss_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    for class_id in focus_class_ids:
        loss_weights[class_id] = float(focus_loss_boost)
    for class_id in partner_class_ids:
        loss_weights[class_id] = max(float(loss_weights[class_id]), float(partner_loss_boost))

    return {
        "sample_weight_map": sample_weight_map,
        "loss_weights": loss_weights,
    }
