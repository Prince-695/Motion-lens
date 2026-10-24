#!/usr/bin/env python3
"""Download sample pre-recorded video clips for testing Chin-Ups and Pull-Ups."""

from pathlib import Path
import urllib.request
import subprocess
import sys

ROOT_DIR = Path(__file__).resolve().parent
CHIN_UPS_INPUT = ROOT_DIR / "chin_ups" / "data" / "input"
PULL_UPS_INPUT = ROOT_DIR / "pull_ups" / "data" / "input"

# Public pre-recorded demonstration clips
SAMPLES = [
    {
        "name": "pullups_sample_1.webm",
        "url": "https://upload.wikimedia.org/wikipedia/commons/1/15/Pull-ups_-_exercise_demonstration_video.webm",
        "target_dirs": [PULL_UPS_INPUT, CHIN_UPS_INPUT],
        "output_filename": "sample_set_1.mp4"
    }
]


def download_and_transcode():
    CHIN_UPS_INPUT.mkdir(parents=True, exist_ok=True)
    PULL_UPS_INPUT.mkdir(parents=True, exist_ok=True)

    for item in SAMPLES:
        temp_file = ROOT_DIR / item["name"]
        print(f"Downloading {item['name']}...")
        try:
            req = urllib.request.Request(
                item["url"],
                headers={"User-Agent": "MotionLens/1.0 (Exercise Analysis Tool)"}
            )
            with urllib.request.urlopen(req) as response, open(temp_file, "wb") as out_file:
                out_file.write(response.read())
            print(f"  Downloaded: {temp_file.stat().st_size / 1024:.1f} KB")

            for target_dir in item["target_dirs"]:
                out_path = target_dir / item["output_filename"]
                print(f"  Transcoding to H.264 MP4: {out_path}...")
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(temp_file), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an", str(out_path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"  Ready: {out_path}")
        except Exception as e:
            print(f"  Error downloading {item['name']}: {e}", file=sys.stderr)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    print("\nPre-recorded sample videos are ready in:")
    print(f"  - {PULL_UPS_INPUT}")
    print(f"  - {CHIN_UPS_INPUT}")


if __name__ == "__main__":
    download_and_transcode()
