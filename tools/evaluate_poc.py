"""Automated Evaluation & Benchmarking for Active Speaker Detection POC."""

import argparse
import json
import os
import time
from pathlib import Path
import pandas as pd
import yaml
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline.cascaded_engine import CascadedEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/poc_manifest.csv")
    parser.add_argument("--config", default="configs/default_poc.yaml")
    parser.add_argument("--out_dir", default="results/evaluation")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    engine = CascadedEngine(cfg)
    manifest = pd.read_csv(args.manifest)

    print("=" * 60)
    print("Running Active Speaker Detection POC Evaluation...")
    print(f"Total samples: {len(manifest)}")
    print("=" * 60)

    results = []
    total_frames = 0
    t0 = time.time()

    for idx, row in tqdm(manifest.iterrows(), total=len(manifest)):
        v_path = row["file_path"]
        cid = row["clip_id"]
        cat = row["category"]
        gt_dec = row["ground_truth_decision"]

        res = engine.process_clip(v_path, clip_id=cid)
        total_frames += len(res.roi_trajectory) if res.roi_trajectory else 75

        res_dict = res.to_dict()
        res_dict["category"] = cat
        res_dict["ground_truth_decision"] = gt_dec
        res_dict["ground_truth_reason"] = row.get("ground_truth_reason", "")
        # Remove deep nested objects from flat summary CSV
        res_dict.pop("timeline", None)
        res_dict.pop("roi_trajectory", None)
        results.append(res_dict)

    elapsed = time.time() - t0
    res_df = pd.DataFrame(results)

    # Save detailed CSV
    res_df.to_csv(out_dir / "evaluation_results.csv", index=False)

    # Compute Metrics
    # In Curation context:
    # 'pass' -> Clean Active Speaker (keep)
    # 'reject' -> Bad / Defective / Voiceover
    # 'manual' -> Ambiguous / Needs Review
    tp = len(res_df[(res_df["ground_truth_decision"] == "keep") & (res_df["decision"] == "pass")])
    fp = len(res_df[(res_df["ground_truth_decision"] == "reject") & (res_df["decision"] == "pass")])
    fn = len(res_df[(res_df["ground_truth_decision"] == "keep") & (res_df["decision"] == "reject")])
    tn = len(res_df[(res_df["ground_truth_decision"] == "reject") & (res_df["decision"] == "reject")])
    manuals = len(res_df[res_df["decision"] == "manual"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Category rejection rates
    cat_stats = {}
    for cat_name, group in res_df.groupby("category"):
        pass_cnt = len(group[group["decision"] == "pass"])
        rej_cnt = len(group[group["decision"] == "reject"])
        man_cnt = len(group[group["decision"] == "manual"])
        cat_stats[cat_name] = {
            "total": len(group),
            "pass": pass_cnt,
            "reject": rej_cnt,
            "manual": man_cnt,
            "rejection_rate": f"{rej_cnt / len(group):.1%}",
        }

    tier_counts = res_df["early_exit_tier"].value_counts().to_dict()

    summary = {
        "total_clips": len(res_df),
        "total_time_seconds": round(elapsed, 2),
        "fps": round(total_frames / elapsed, 1) if elapsed > 0 else 0.0,
        "seconds_per_clip": round(elapsed / len(res_df), 3) if len(res_df) > 0 else 0.0,
        "metrics": {
            "clean_speaker_precision": round(precision, 4),
            "clean_speaker_recall": round(recall, 4),
            "clean_speaker_f1": round(f1, 4),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "sent_to_manual_review": manuals,
        },
        "by_category": cat_stats,
        "early_exit_breakdown": {f"tier_{k}": v for k, v in tier_counts.items()},
    }

    with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDetailed outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
