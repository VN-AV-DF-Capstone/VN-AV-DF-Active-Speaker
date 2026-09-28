"""Download pretrained model weights for Active Speaker Detection POC."""

import hashlib
import os
import urllib.request
from pathlib import Path


WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "weights"

MODELS = {
    "silero_vad.onnx": {
        "url": "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx",
        "size_mb": 2.0,
        "description": "Silero VAD v5 official 16kHz ONNX model",
    },
    "light_asd.pth": {
        "url": "https://github.com/Junhua-Liao/Light-ASD/releases/download/v1.0/light_asd_ava.pth",
        "size_mb": 4.5,
        "description": "Light-ASD official checkpoint on AVA-ActiveSpeaker",
    },
}


def download_file(url: str, dest_path: Path):
    print(f"Downloading {dest_path.name} from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"  -> Successfully saved to {dest_path}")
    except Exception as e:
        print(f"  -> Warning: Automated download failed ({e}).")
        print(f"     Please manually download from: {url}")
        print(f"     and place it at: {dest_path}")


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("Active Speaker Detection Weight Downloader")
    print(f"Target directory: {WEIGHTS_DIR}")
    print("=" * 60)

    for filename, info in MODELS.items():
        dest = WEIGHTS_DIR / filename
        if dest.is_file():
            print(f"[OK] {filename} already exists ({dest.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            download_file(info["url"], dest)

    print("\nWeight check completed.")


if __name__ == "__main__":
    main()
