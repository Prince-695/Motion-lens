# Motion-Lens

AI-powered biomechanical video analysis for bodyweight training, specialized in **Chin-Ups** and **Pull-Ups**.

Motion-Lens extracts keypoint telemetry from video, tracks joint displacement relative to the bar, measures concentric (ascent) and eccentric (descent) phases, and grades range of motion (ROM) against calibrated dead hangs.

---

## Supported Exercises

| Exercise | Key Capabilities | Key Model | Directory |
|---|---|---|---|
| **Chin-Ups** | Rep counting, concentric ascent & eccentric descent timing, peak displacement analysis | `vitpose-plus-large` via VLM Run Gateway | [`chin_ups/`](chin_ups/) |
| **Pull-Ups** | Rep counting, range of motion grading (Full vs. Partial), elbow angle tracking, athlete dead-hang calibration | `vitpose-plus-large` via VLM Run Gateway | [`pull_ups/`](pull_ups/) |

---

## Quickstart

### 1. Environment Setup

A Python 3.11 virtual environment with static `ffmpeg` and `ffprobe` is configured in `.venv`:

```bash
# Activate the environment
source .venv/bin/activate
```

Dependencies include:
- `opencv-python`, `numpy`, `matplotlib`, `pillow`
- `openai>=1.40`, `python-dotenv`, `rich`
- `static-ffmpeg` (with `ffmpeg` and `ffprobe` linked to `~/.local/bin`)

### 2. Configure VLM Run Gateway API Key

Get an API key at [app.vlm.run](https://app.vlm.run) and place it in the root `.env`:

```bash
# .env
VLMRUN_API_KEY=your_actual_vlm_run_api_key_here
```

### 3. Add Input Videos

Place pre-recorded video clips (`.mp4`, `.mov`, `.m4v`, `.webm`) into the respective input directory:

- **Chin-ups**: `chin_ups/data/input/`
- **Pull-ups**: `pull_ups/data/input/`

Or fetch sample pre-recorded test clips automatically:
```bash
.venv/bin/python download_samples.py
```

---

## Running the Analyses

### Chin-Ups
```bash
cd chin_ups
../.venv/bin/python main.py
```
Options:
- `--trim 3`: Fast test on only the first 3 seconds of the clip.
- `--height 1080` (or `--height full`): Controls export resolution.
- `--li "Your Name"` / `--x @handle`: Custom watermark overlay.

### Pull-Ups
```bash
cd pull_ups
# Run all clips in data/input/
../.venv/bin/python main.py

# Or run a specific clip
../.venv/bin/python main.py data/input/demo_pullups.mp4

# Calibrate against a set of full reps
../.venv/bin/python calibrate.py data/input/my_full_set.mp4
```

---

## Datasets for Testing

Pre-recorded video datasets available online:
1. **[RepCount Benchmark](https://github.com/Yuxuan-Chen-99/RepCount)**: Academic repetition counting dataset containing hundreds of pull-up and chin-up videos with frame-accurate start/peak/end annotations.
2. **[Countix Dataset](https://deepmind.google/discover/blog/countix-a-repetition-counting-dataset/)**: Repetition action video dataset covering pull-up exercise sequences.
3. **Pexels & Wikimedia Commons**: Free-to-use CC0 / CC-BY exercise clips filmed with clear bar visibility.

---

## License

[Apache-2.0](LICENSE)
