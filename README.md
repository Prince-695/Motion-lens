# Motion-Lens

AI-powered biomechanical video analysis for bodyweight training, specialized in **Chin-Ups** and **Pull-Ups**.

Motion-Lens extracts keypoint telemetry from pre-recorded videos, tracks joint displacement relative to the bar, measures concentric (ascent) and eccentric (descent) phases, grades range of motion (ROM) against calibrated dead hangs, and renders a side-by-side telemetry panel in a **real-time interactive window**.

---

## Key Features

- **Real-Time Interactive Window**: Plays the original video alongside the live skeleton overlay, elbow angle arcs, rep telemetry, and timing graph without forcing disk saves.
- **Playback Controls**:
  - `Space`: Pause / Play
  - `Q` or `ESC`: Quit playback
- **100% Offline by Default**: Powered by a local YOLOv8-pose ONNX engine (`models/yolov8n-pose.onnx`) running on CPU/GPU without cloud API keys.
- **Optional Video Export**: Keep disk usage light by default (`--no-save`). Save rendered high-definition MP4 files on demand with `--save`.
- **Pre-recorded Datasets Ready**: Works with any `.mp4`, `.mov`, `.m4v`, or `.webm` files from phone recordings, gym cameras, or benchmark datasets (RepCount, Countix, Pexels).

---

## Supported Exercises

| Exercise | Key Capabilities | Default Pose Model | Directory |
|---|---|---|---|
| **Chin-Ups** | Rep counting, concentric ascent & eccentric descent timing, peak displacement analysis, side-by-side stats HUD | Local YOLOv8-pose ONNX (offline) | [`chin_ups/`](chin_ups/) |
| **Pull-Ups** | Rep counting, range of motion grading (Full vs. Partial), live elbow angle arcs, athlete dead-hang calibration | Local YOLOv8-pose ONNX (offline) | [`pull_ups/`](pull_ups/) |

---

## Quickstart

### 1. Environment Setup

A Python 3.11 virtual environment with all dependencies and static `ffmpeg`/`ffprobe` is configured in `.venv`:

```bash
# Activate the environment
source .venv/bin/activate
```

Dependencies include:
- `opencv-python`, `onnxruntime`, `numpy`, `matplotlib`, `pillow`, `rich`, `python-dotenv`
- `ffmpeg` and `ffprobe` static binaries

### 2. Sample Datasets

Sample clips are stored in `chin_ups/data/input/` and `pull_ups/data/input/`. You can download sample test clips or drop your own:

```bash
.venv/bin/python download_samples.py
```

You can place any pre-recorded workout videos in:
- `chin_ups/data/input/`
- `pull_ups/data/input/`
- Or pass the video path directly on the command line!

---

## Running Motion-Lens in Real Time

### Pull-Ups

```bash
cd pull_ups

# Run the default sample with real-time interactive window:
../.venv/bin/python main.py

# Or run any specific video file:
../.venv/bin/python main.py data/input/sample_set_1.mp4
../.venv/bin/python main.py /path/to/my_pullup_video.mp4

# Run and also save the rendered MP4 to data/output/
../.venv/bin/python main.py --save

# Calibrate against a set of full reps
../.venv/bin/python calibrate.py data/input/sample_set_1.mp4
```

### Chin-Ups

```bash
cd chin_ups

# Run with real-time interactive window:
../.venv/bin/python main.py

# Or run any specific video file:
../.venv/bin/python main.py data/input/sample_set_1.mp4
../.venv/bin/python main.py /path/to/my_chinup_video.mp4

# Run and also save the rendered MP4 to data/output/
../.venv/bin/python main.py --save
```

---

## Real-Time Window & CLI Controls

| Key / Flag | Description |
|---|---|
| `Space` | Pause / Resume playback in the display window |
| `Q` / `ESC` | Exit the window and stop playback |
| `--show` / `--no-show` | Show or suppress the interactive OpenCV GUI window (default: `--show`) |
| `--save` / `--no-save` | Save or skip saving the rendered `.mp4` video to `data/output/` (default: `--no-save`) |
| `--local` | Use the local offline YOLOv8-pose ONNX model (default) |
| `--gateway` | Use cloud VLM Run gateway (`vitpose-plus-large`) if `VLMRUN_API_KEY` is set in `.env` |
| `--height <px>` | Scaling height (`720`, `1080`, or `full`) |
| `--trim <sec>` | Only process the first N seconds of the video |

---

## Recommended Video Datasets for Testing

Pre-recorded video datasets available online:
1. **[RepCount Benchmark](https://github.com/Yuxuan-Chen-99/RepCount)**: Benchmark containing hundreds of annotated pull-up and chin-up workout videos across diverse camera angles and gym environments.
2. **[Countix Dataset](https://deepmind.google/discover/blog/countix-a-repetition-counting-dataset/)**: Large-scale repetition action dataset featuring pull-up and chin-up sequences.
3. **Pexels & Wikimedia Commons**: Free-to-use CC0 / CC-BY fitness videos with clear visibility of athlete, bar, and full range of motion.

---

## License

[Apache-2.0](LICENSE)
