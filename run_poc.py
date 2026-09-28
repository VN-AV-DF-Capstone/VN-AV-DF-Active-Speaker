"""Active Speaker Detection & ROI Tracking POC Runner.

Run single clip inference or batch processing.
Usage examples:
    # Single clip with visualization:
    python run_poc.py --video "test.mp4" --visualize

    # Batch processing from manifest:
    python run_poc.py --manifest "data/poc_manifest.csv" --out_dir "results"
"""

import argparse
import json
import os
import sys
from pathlib import Path
import pandas as pd
import yaml

from src.pipeline.cascaded_engine import CascadedEngine
from src.utils.audio_video import extract_audio_16k_mono
from src.utils.dashboard import generate_diagnostic_dashboard
from src.utils.visualizer import render_annotated_video


def process_single(engine: CascadedEngine, video_path: str, out_dir: Path, visualize: bool):
    print(f"\n[Processing Clip] {video_path}")
    cid = Path(video_path).stem
    result = engine.process_clip(video_path, clip_id=cid)

    # Print summary
    print(f"  -> Decision: {result.decision.upper()}" + (f" (Reason: {result.reason})" if result.reason else ""))
    print(f"  -> Voiced Speech: {result.voiced_ms} ms | Active Speech Ratio: {result.visible_active_speech_ratio:.1%}")
    print(f"  -> Primary Speaker Track: {result.primary_speaker_track_id} | Early-Exit Tier: {result.early_exit_tier}")

    clip_out_dir = out_dir / cid
    clip_out_dir.mkdir(parents=True, exist_ok=True)

    # Write JSON metadata
    res_path = clip_out_dir / "result.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"  -> Saved metadata: {res_path}")

    # Write ROI trajectory
    roi_path = clip_out_dir / "active_roi.json"
    with open(roi_path, "w", encoding="utf-8") as f:
        json.dump({
            "clip_id": cid,
            "primary_speaker_track_id": result.primary_speaker_track_id,
            "roi_trajectory": result.roi_trajectory,
        }, f, indent=2, ensure_ascii=False)
    print(f"  -> Saved ROI trajectory: {roi_path}")

    # Render video and diagnostic dashboard if requested
    if visualize:
        vis_path = clip_out_dir / f"{cid}_annotated.mp4"
        print("  -> Rendering annotated visualization video...")
        render_annotated_video(video_path, result, str(vis_path))
        print(f"  -> Visualized video: {vis_path}")

        dash_path = clip_out_dir / f"{cid}_diagnostic_dashboard.png"
        print("  -> Rendering Mel-Spectrogram & Audiovisual Diagnostic Dashboard...")
        audio = extract_audio_16k_mono(video_path)
        generate_diagnostic_dashboard(audio, result, str(dash_path))
        print(f"  -> Diagnostic dashboard: {dash_path}")


def main():
    parser = argparse.ArgumentParser(description="Active Speaker & ROI Tracking POC")
    parser.add_argument("--video", default=None, help="Path to a single video clip")
    parser.add_argument("--manifest", default=None, help="Path to manifest CSV")
    parser.add_argument("--config", default="configs/default_poc.yaml", help="Path to YAML config")
    parser.add_argument("--out_dir", default="results", help="Directory for output results")
    parser.add_argument("--visualize", action="store_true", help="Render annotated visualization video")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of manifest clips to process")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    engine = CascadedEngine(cfg)

    if args.video:
        process_single(engine, args.video, out_dir, args.visualize)
    elif args.manifest:
        df = pd.read_csv(args.manifest)
        if args.limit:
            df = df.head(args.limit)
        print(f"Processing batch of {len(df)} clips from {args.manifest}...")
        for _, row in df.iterrows():
            process_single(engine, row["file_path"], out_dir, args.visualize)
    else:
        print("Error: Please provide either --video or --manifest")
        sys.exit(1)


if __name__ == "__main__":
    main()
