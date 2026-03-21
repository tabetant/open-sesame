"""
export_weights.py
Exports trained Keras model weights + all precomputed MFCC tables into two C files:
  output/model_data.h  — extern declarations
  output/model_data.c  — actual array definitions

Run once after training. The generated files are #included / compiled into the
Nios V project alongside mfcc.c and inference.c.
"""

import os
import numpy as np
import librosa
import tensorflow as tf
from scipy.fftpack import dct as scipy_dct

SAMPLE_RATE = 8000
N_FFT       = 200
HOP_LENGTH  = 160
N_MELS      = 26
N_MFCC      = 13
N_FRAMES    = 99

OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output")
MODEL_PATH  = os.path.join(OUTPUT_DIR, "voice_gate.keras")
NORM_PATH   = os.path.join(OUTPUT_DIR, "normalization.txt")
HEADER_PATH = os.path.join(OUTPUT_DIR, "model_data.h")
SOURCE_PATH = os.path.join(OUTPUT_DIR, "model_data.c")

# ── Load model and normalization ──────────────────────────────────────────────
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

with open(NORM_PATH) as f:
    lines = f.read().strip().split()
    mfcc_mean = float(lines[0])
    mfcc_std  = float(lines[1])
print(f"Normalization: mean={mfcc_mean:.8f}, std={mfcc_std:.8f}")


# ── Verify flatten size matches Dense weight ──────────────────────────────────
# After Conv1(same,13x99x32) → Pool(2x2,6x49x32) → Conv2(same,6x49x64)
# → Pool(2x2,3x24x64) → Flatten = 3*24*64 = 4608
flatten_size = 3 * 24 * 64
assert model.get_layer('dense').get_weights()[0].shape[0] == flatten_size, \
    f"Flatten mismatch: expected {flatten_size}"
print(f"Flatten size confirmed: {flatten_size}")


# ── Helper: array → C literal ─────────────────────────────────────────────────
def to_c_array(arr, name, type_str="float"):
    flat = arr.flatten().tolist()
    shape_str = "".join(f"[{d}]" for d in arr.shape)
    lines = [f"const {type_str} {name}{shape_str} = {{"]
    per_row = 8
    for i in range(0, len(flat), per_row):
        chunk = flat[i:i+per_row]
        row = ", ".join(f"{v:.8e}f" for v in chunk)
        lines.append(f"    {row},")
    lines.append("};")
    return "\n".join(lines)


# ── Extract CNN weights ───────────────────────────────────────────────────────
# Keras Conv2D weight layout: (kH, kW, in_ch, out_ch)
# C inference loop layout:    [out_ch][in_ch][kH][kW]  (transposed for cache)

print("Extracting CNN weights...")

conv1_w_keras, conv1_b = model.get_layer('conv2d').get_weights()
conv1_w = np.transpose(conv1_w_keras, (3, 2, 0, 1)).astype(np.float32)
# shape: (32, 1, 3, 3)

conv2_w_keras, conv2_b = model.get_layer('conv2d_1').get_weights()
conv2_w = np.transpose(conv2_w_keras, (3, 2, 0, 1)).astype(np.float32)
# shape: (64, 32, 3, 3)

# Keras Dense layout: (in_features, out_features) — matches C loop directly
dense1_w, dense1_b = model.get_layer('dense').get_weights()
dense1_w = dense1_w.astype(np.float32)   # (4608, 64)

dense2_w, dense2_b = model.get_layer('dense_1').get_weights()
dense2_w = dense2_w.astype(np.float32)   # (64, 2)

conv1_b  = conv1_b.astype(np.float32)
conv2_b  = conv2_b.astype(np.float32)
dense1_b = dense1_b.astype(np.float32)
dense2_b = dense2_b.astype(np.float32)

print(f"  conv1_w: {conv1_w.shape}  conv1_b: {conv1_b.shape}")
print(f"  conv2_w: {conv2_w.shape}  conv2_b: {conv2_b.shape}")
print(f"  dense1_w: {dense1_w.shape}  dense1_b: {dense1_b.shape}")
print(f"  dense2_w: {dense2_w.shape}  dense2_b: {dense2_b.shape}")


# ── Precompute MFCC tables ────────────────────────────────────────────────────
# All trig computed here; C does only multiply-accumulate at runtime.

print("Precomputing MFCC tables...")

# Hann window (librosa default — same as scipy.signal.hann)
hann_win = np.hanning(N_FFT).astype(np.float32)    # shape: (200,)

# DFT twiddle matrix: cos_table[k][n] = cos(2*pi*k*n/N_FFT)
#                     sin_table[k][n] = sin(2*pi*k*n/N_FFT)
# Only positive frequencies: k = 0 .. N_FFT//2  (101 bins)
N_BINS = N_FFT // 2 + 1   # 101
n_idx  = np.arange(N_FFT, dtype=np.float64)
k_idx  = np.arange(N_BINS, dtype=np.float64)
phase  = 2.0 * np.pi * np.outer(k_idx, n_idx) / N_FFT
cos_table = np.cos(phase).astype(np.float32)   # (101, 200)
sin_table = np.sin(phase).astype(np.float32)   # (101, 200)

# Mel filterbank: shape (N_MELS, N_BINS) = (26, 101)
mel_fb = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=N_MELS).astype(np.float32)
# shape: (26, 101)

# DCT-II matrix: shape (N_MFCC, N_MELS) = (13, 26)
# Matches scipy.fftpack.dct(type=2, norm='ortho') which librosa uses internally.
dct_mat = np.zeros((N_MFCC, N_MELS), dtype=np.float32)
for i in range(N_MFCC):
    for j in range(N_MELS):
        dct_mat[i, j] = np.cos(np.pi * i * (j + 0.5) / N_MELS)
# Apply ortho normalization to match scipy dct norm='ortho'
dct_mat[0, :] *= np.sqrt(1.0 / N_MELS)
dct_mat[1:, :] *= np.sqrt(2.0 / N_MELS)
dct_mat = dct_mat.astype(np.float32)

print(f"  hann_win:  {hann_win.shape}")
print(f"  cos_table: {cos_table.shape}")
print(f"  sin_table: {sin_table.shape}")
print(f"  mel_fb:    {mel_fb.shape}")
print(f"  dct_mat:   {dct_mat.shape}")

# ── Sanity check: compare C-style MFCC to librosa on a test signal ───────────
print("\nRunning MFCC sanity check...")
test_audio = np.random.RandomState(0).randn(16000).astype(np.float32) * 0.1

# librosa reference (center=False)
ref_mfcc = librosa.feature.mfcc(
    y=test_audio, sr=SAMPLE_RATE,
    n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    center=False
)[:, :N_FRAMES]  # (13, 99)

# Simulate C computation (using exact same formula as librosa):
#   librosa.feature.mfcc calls power_to_db(melspectrogram(...))
#   power_to_db: 10*log10(max(1e-10, S)), then clip to [max - 80, max]
AMIN    = 1e-10
TOP_DB  = 80.0

log_mel_all = np.zeros((N_MELS, N_FRAMES), dtype=np.float64)
for t in range(N_FRAMES):
    start = t * HOP_LENGTH
    frame = test_audio[start:start + N_FFT].astype(np.float64) * hann_win.astype(np.float64)
    # DFT
    real  = cos_table.astype(np.float64) @ frame   # (101,)
    imag  = sin_table.astype(np.float64) @ frame   # (101,)
    power = real**2 + imag**2                      # (101,)
    # Mel filterbank
    mel_e = mel_fb.astype(np.float64) @ power      # (26,)
    # power_to_db (matching librosa with ref=1.0, amin=1e-10)
    log_mel_all[:, t] = 10.0 * np.log10(np.maximum(AMIN, mel_e))

# Apply top_db clipping (global max over all frames and mel bands)
global_max = log_mel_all.max()
log_mel_all = np.maximum(log_mel_all, global_max - TOP_DB)

# Apply DCT per frame
sim_mfcc = np.zeros((N_MFCC, N_FRAMES), dtype=np.float64)
for t in range(N_FRAMES):
    sim_mfcc[:, t] = dct_mat.astype(np.float64) @ log_mel_all[:, t]

max_err = np.max(np.abs(sim_mfcc - ref_mfcc))
mean_err = np.mean(np.abs(sim_mfcc - ref_mfcc))
print(f"  Max absolute error vs librosa: {max_err:.6f}")
print(f"  Mean absolute error vs librosa: {mean_err:.6f}")
if max_err > 1.0:
    print("  WARNING: large MFCC mismatch — check twiddle/DCT tables")
else:
    print("  MFCC tables validated OK")


# ── Write model_data.h ────────────────────────────────────────────────────────
print("\nWriting model_data.h...")

header = f"""/* model_data.h — AUTO-GENERATED by export_weights.py. DO NOT EDIT.
 * Contains extern declarations for all CNN weights and MFCC precomputed tables.
 * Include this header in mfcc.c and inference.c.
 */
#ifndef MODEL_DATA_H
#define MODEL_DATA_H

/* ── MFCC constants ─────────────────────────────────────────────────────── */
#define SAMPLE_RATE   {SAMPLE_RATE}
#define N_FFT         {N_FFT}
#define HOP_LENGTH    {HOP_LENGTH}
#define N_MELS        {N_MELS}
#define N_MFCC        {N_MFCC}
#define N_FRAMES      {N_FRAMES}
#define N_BINS        {N_BINS}      /* N_FFT/2 + 1 */
#define MFCC_MEAN     {mfcc_mean:.8e}f
#define MFCC_STD      {mfcc_std:.8e}f
#define LOG_FLOOR     1e-6f

/* ── MFCC precomputed tables ─────────────────────────────────────────────── */
extern const float hann_win[{N_FFT}];
extern const float cos_table[{N_BINS}][{N_FFT}];
extern const float sin_table[{N_BINS}][{N_FFT}];
extern const float mel_fb[{N_MELS}][{N_BINS}];
extern const float dct_mat[{N_MFCC}][{N_MELS}];

/* ── CNN weights ─────────────────────────────────────────────────────────── */
/* Conv1: 32 filters, 1 input channel, 3x3 kernel */
extern const float conv1_w[32][1][3][3];
extern const float conv1_b[32];

/* Conv2: 64 filters, 32 input channels, 3x3 kernel */
extern const float conv2_w[64][32][3][3];
extern const float conv2_b[64];

/* Dense1: 4608 inputs → 64 outputs */
extern const float dense1_w[4608][64];
extern const float dense1_b[64];

/* Dense2: 64 inputs → 2 outputs */
extern const float dense2_w[64][2];
extern const float dense2_b[2];

#endif /* MODEL_DATA_H */
"""

with open(HEADER_PATH, "w") as f:
    f.write(header)
print(f"  Written: {HEADER_PATH}")


# ── Write model_data.c ────────────────────────────────────────────────────────
print("Writing model_data.c  (this may take a minute — 295K floats for dense1)...")

sections = [
    '/* model_data.c — AUTO-GENERATED by export_weights.py. DO NOT EDIT. */',
    '#include "model_data.h"',
    '',
    '/* ── MFCC tables ──────────────────────────────────────────────────── */',
    to_c_array(hann_win,   "hann_win"),
    to_c_array(cos_table,  "cos_table"),
    to_c_array(sin_table,  "sin_table"),
    to_c_array(mel_fb,     "mel_fb"),
    to_c_array(dct_mat,    "dct_mat"),
    '',
    '/* ── CNN weights ──────────────────────────────────────────────────── */',
    to_c_array(conv1_w,  "conv1_w"),
    to_c_array(conv1_b,  "conv1_b"),
    to_c_array(conv2_w,  "conv2_w"),
    to_c_array(conv2_b,  "conv2_b"),
    to_c_array(dense1_w, "dense1_w"),
    to_c_array(dense1_b, "dense1_b"),
    to_c_array(dense2_w, "dense2_w"),
    to_c_array(dense2_b, "dense2_b"),
]

with open(SOURCE_PATH, "w") as f:
    f.write("\n".join(sections) + "\n")

size_mb = os.path.getsize(SOURCE_PATH) / (1024 * 1024)
print(f"  Written: {SOURCE_PATH}  ({size_mb:.1f} MB source)")
print("\nExport complete. Files ready for Nios V project:")
print(f"  {HEADER_PATH}")
print(f"  {SOURCE_PATH}")
