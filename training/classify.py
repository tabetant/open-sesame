"""
classify.py — Run inference on one or more audio files using the trained voice gate model.

Usage:
    python classify.py path/to/audio.wav [more files ...]
    python classify.py path/to/folder/         # classifies all .wav files in folder

Examples:
    python classify.py my_recording.wav
    python classify.py ../training/audiosamples/negative_speech_commands/right/*.wav
"""

import os
import sys
import glob
import numpy as np
import librosa
import tensorflow as tf

# ── Constants (must match train_voice_gate.py) ────────────────────────────────
SAMPLE_RATE   = 8000
TARGET_LENGTH = 16000
N_MFCC        = 13
N_FFT         = 200
HOP_LENGTH    = 160
N_MELS        = 26
N_FRAMES      = 99

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "voice_gate.keras")
NORM_PATH   = os.path.join(OUTPUT_DIR, "normalization.txt")


NOISE_FLOOR = 0.001


def extract_mfcc(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < TARGET_LENGTH:
        pad_len = TARGET_LENGTH - len(audio)
        buf = np.random.randn(TARGET_LENGTH).astype(np.float32) * NOISE_FLOOR
        buf[:len(audio)] = audio
        audio = buf
    else:
        audio = audio[:TARGET_LENGTH]
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        center=False
    )
    if mfcc.shape[1] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :N_FRAMES]
    return mfcc[..., np.newaxis]  # (13, 99, 1)


def main():
    # ── Collect input files ───────────────────────────────────────────────────
    if len(sys.argv) < 2:
        print("Usage: python classify.py <audio.wav> [more files or folder]")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            for ext in ("*.wav", "*.m4a"):
                paths.extend(sorted(glob.glob(os.path.join(arg, ext))))
        elif os.path.exists(arg):
            paths.append(arg)
        else:
            paths.extend(sorted(glob.glob(arg)))

    if not paths:
        print("No .wav files found.")
        sys.exit(1)

    # ── Load model + normalization ────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"Model not found: {MODEL_PATH}")
        print("Run train_voice_gate.py first.")
        sys.exit(1)

    print(f"Loading model from {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)

    with open(NORM_PATH) as f:
        lines = f.read().strip().split()
    mean = float(lines[0])
    std  = float(lines[1])
    print(f"Normalization: mean={mean:.6f}, std={std:.6f}\n")

    # ── Run inference ─────────────────────────────────────────────────────────
    label_names = {0: "NEGATIVE", 1: "POSITIVE (open sesame)"}
    col_w = max(len(p) for p in paths) + 2

    print(f"{'File':<{col_w}}  {'Label':<26}  Confidence")
    print("-" * (col_w + 40))

    results = []
    for path in paths:
        try:
            feat = extract_mfcc(path)
            feat = (feat - mean) / std
            feat = feat[np.newaxis, ...]          # (1, 13, 99, 1)
            probs = model.predict(feat, verbose=0)[0]  # (2,)
            pred  = int(np.argmax(probs))
            conf  = float(probs[pred])
            results.append((path, pred, conf, probs))
            label = label_names[pred]
            print(f"{path:<{col_w}}  {label:<26}  {conf*100:.1f}%")
        except Exception as e:
            print(f"{path:<{col_w}}  ERROR: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if results:
        n_pos = sum(1 for _, p, _, _ in results if p == 1)
        n_neg = sum(1 for _, p, _, _ in results if p == 0)
        print(f"\nSummary: {len(results)} files — {n_pos} POSITIVE, {n_neg} NEGATIVE")

        high_conf_pos = [(path, conf) for path, pred, conf, _ in results if pred == 1 and conf > 0.9]
        if high_conf_pos:
            print(f"High-confidence positives (>90%): {len(high_conf_pos)}")


if __name__ == "__main__":
    main()
