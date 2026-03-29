"""
prepare_data.py — Convert DE1-SoC raw audio CSV dumps into .wav files.

The DE1-SoC audio recorder prints 32-bit signed integers via JTAG UART,
bracketed by CLIP_START/CLIP_END markers.  This script parses those files,
converts to 16-bit PCM .wav at 8 kHz, and writes into dataset/positive/
and dataset/negative/ for use by train_voice_gate.py.

Usage:
    python prepare_data.py
"""

import os
import re
import numpy as np
import soundfile as sf

SAMPLE_RATE = 8000
EXPECTED_SAMPLES = SAMPLE_RATE * 5   # 40000 per clip (5 seconds)

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
POS_DIR     = os.path.join(DATASET_DIR, "positive")
NEG_DIR     = os.path.join(DATASET_DIR, "negative")

POS_CSV = os.path.join(os.path.expanduser("~"), "Downloads", "pos_data.csv")
NEG_CSV = os.path.join(os.path.expanduser("~"), "Downloads", "negative_data_1.csv")


def parse_csv(filepath):
    """Extract clip arrays from a DE1-SoC audio dump CSV."""
    clips = {}
    current_clip = None
    current_samples = []

    with open(filepath, encoding="utf-16") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            start = re.match(r"CLIP_START\s+(\d+)", line)
            if start:
                current_clip = int(start.group(1))
                current_samples = []
                continue

            end = re.match(r"CLIP_END\s+(\d+)", line)
            if end:
                clip_num = int(end.group(1))
                if current_clip == clip_num and current_samples:
                    clips[clip_num] = np.array(current_samples, dtype=np.int64)
                current_clip = None
                current_samples = []
                continue

            if current_clip is None:
                continue

            try:
                vals = [int(v) for v in line.split(",") if v.strip().lstrip("-").isdigit()]
                current_samples.extend(vals)
            except ValueError:
                continue

    return clips


def convert_and_save(clips, out_dir, prefix):
    """Convert raw 32-bit samples to float, remove DC offset, save as .wav."""
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for clip_num in sorted(clips.keys()):
        raw = clips[clip_num]
        if len(raw) < EXPECTED_SAMPLES * 0.9:
            print(f"  Skipping {prefix}_{clip_num:03d} — only {len(raw)} samples")
            continue
        audio = raw.astype(np.float64) / 2147483648.0
        audio -= np.mean(audio)
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        path = os.path.join(out_dir, f"{prefix}_{clip_num:03d}.wav")
        sf.write(path, audio, SAMPLE_RATE)
        saved += 1
    return saved


def main():
    print("Preparing dataset from DE1-SoC recordings...\n")

    print(f"Parsing positive data: {POS_CSV}")
    pos_clips = parse_csv(POS_CSV)
    print(f"  Found {len(pos_clips)} complete clips")
    n_pos = convert_and_save(pos_clips, POS_DIR, "pos")
    print(f"  Saved {n_pos} positive .wav files → {POS_DIR}\n")

    print(f"Parsing negative data: {NEG_CSV}")
    neg_clips = parse_csv(NEG_CSV)
    print(f"  Found {len(neg_clips)} complete clips")
    n_neg = convert_and_save(neg_clips, NEG_DIR, "neg")
    print(f"  Saved {n_neg} negative .wav files → {NEG_DIR}\n")

    sample_clip = list(pos_clips.values())[0] if pos_clips else list(neg_clips.values())[0]
    audio = sample_clip.astype(np.float64) / 2147483648.0
    audio -= np.mean(audio)
    print(f"Sample clip stats: {len(sample_clip)} samples, "
          f"range [{audio.min():.4f}, {audio.max():.4f}], "
          f"rms={np.sqrt(np.mean(audio**2)):.4f}")

    print(f"\nDataset ready: {n_pos} positive, {n_neg} negative")
    print("Run train_voice_gate.py next.")


if __name__ == "__main__":
    main()
