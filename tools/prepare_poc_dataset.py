"""Prepare a balanced, categorized mini-dataset for Active Speaker POC.

Scans human reviews in VN-AV-DF-Capstone to assemble:
1. Clean Single Speaker (keep)
2. Mouth Movement Defect / Freeze (reject - mouth)
3. Voiceover / B-roll (reject - voiceover)
4. Multi-Face / Wrong Face Competition (reject - wrong_face)
"""

import argparse
import os
from pathlib import Path
import pandas as pd


DEFAULT_REVIEW_CSV = (
    r"L:\FPT\Side project\Capstone\resources\VN-AV-DF-Capstone"
    r"\data\manifests\dataset_v1\reviews\exports\clips\nguyenlamanh\review_nguyenlamanh.csv"
)
DEFAULT_CLIPS_DIR = (
    r"L:\FPT\Side project\Capstone\resources\VN-AV-DF-Capstone"
    r"\data\manifests\dataset_v1\reviews\exports\clips\nguyenlamanh"
)
DEFAULT_OUT_CSV = (
    r"L:\FPT\Side project\Capstone\resources\VN-AV-DF-Active-Speaker\data\poc_manifest.csv"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review_csv", default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--clips_dir", default=DEFAULT_CLIPS_DIR)
    parser.add_argument("--out_csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--samples_per_cat", type=int, default=15)
    args = parser.parse_args()

    if not os.path.isfile(args.review_csv):
        raise FileNotFoundError(f"Review CSV not found: {args.review_csv}")

    df = pd.read_csv(args.review_csv)
    clips_dir = Path(args.clips_dir)

    # Resolve full file paths
    df["full_path"] = [str(clips_dir / Path(fp).name) for fp in df["file_path"]]
    df["exists"] = [os.path.isfile(p) for p in df["full_path"]]
    valid_df = df[df["exists"]].copy()
    print(f"Total reviewed clips with existing media: {len(valid_df)}")

    selected = []

    # Category 1: Clean Single Speaker (keep)
    keeps = valid_df[valid_df["decision"] == "keep"].sample(
        min(args.samples_per_cat, len(valid_df[valid_df["decision"] == "keep"])),
        random_state=42
    ).copy()
    keeps["category"] = "clean_single_speaker"
    selected.append(keeps)

    # Category 2: Mouth Defect / Lip Desync (reject - mouth)
    mouths = valid_df[
        (valid_df["decision"] == "reject") &
        (valid_df["reason"].fillna("").str.contains("mouth"))
    ].sample(
        min(args.samples_per_cat, len(valid_df[valid_df["reason"].fillna("").str.contains("mouth")])),
        random_state=42
    ).copy()
    mouths["category"] = "mouth_defect_or_freeze"
    selected.append(mouths)

    # Category 3: Voiceover (all available)
    voiceovers = valid_df[
        (valid_df["decision"] == "reject") &
        (valid_df["reason"].fillna("").str.contains("voiceover"))
    ].copy()
    voiceovers["category"] = "voiceover_broll"
    selected.append(voiceovers)

    # Category 4: Multi-Face / Wrong Face (all available)
    wrong_faces = valid_df[
        (valid_df["decision"] == "reject") &
        (valid_df["reason"].fillna("").str.contains("wrong_face"))
    ].copy()
    wrong_faces["category"] = "multi_face_competition"
    selected.append(wrong_faces)

    poc_df = pd.concat(selected, ignore_index=True)
    out_cols = {
        "clip_id": "clip_id",
        "full_path": "file_path",
        "category": "category",
        "decision": "ground_truth_decision",
        "reason": "ground_truth_reason",
    }
    poc_manifest = poc_df[list(out_cols.keys())].rename(columns=out_cols)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    poc_manifest.to_csv(out_path, index=False)

    print("=" * 60)
    print(f"POC Manifest Generated: {out_path}")
    print(f"Total samples: {len(poc_manifest)}")
    print("Breakdown by category:")
    print(poc_manifest["category"].value_counts())
    print("=" * 60)


if __name__ == "__main__":
    main()
