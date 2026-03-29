import os
import random
import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix

# ── Constants (must match C implementation on Nios V) ─────────────────────────
SAMPLE_RATE    = 8000
TARGET_LENGTH  = 16000   # 2 seconds
N_MFCC         = 13
N_FFT          = 200
HOP_LENGTH     = 160
N_MELS         = 26
N_FRAMES       = 99
BATCH_SIZE     = 16
EPOCHS         = 50
AUGMENT_COPIES = 4

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

NOISE_FLOOR = 0.001


# ── Audio preprocessing ───────────────────────────────────────────────────────

def _random_crop(audio, target_len):
    """Randomly crop or noise-pad audio to exactly target_len samples."""
    if len(audio) >= target_len:
        start = np.random.randint(0, len(audio) - target_len + 1)
        return audio[start:start + target_len].copy()
    pad_total = target_len - len(audio)
    offset = np.random.randint(0, pad_total + 1)
    buf = np.random.randn(target_len).astype(np.float32) * NOISE_FLOOR
    buf[offset:offset + len(audio)] = audio
    return buf


def _energy_crop(audio, target_len, jitter=0):
    """Crop target_len samples centered on the loudest region.
    Use jitter > 0 for augmentation (random offset from peak energy)."""
    if len(audio) <= target_len:
        pad_total = target_len - len(audio)
        offset = pad_total // 2
        buf = np.random.randn(target_len).astype(np.float32) * NOISE_FLOOR
        buf[offset:offset + len(audio)] = audio
        return buf

    sq = audio.astype(np.float64) ** 2
    cumsum = np.concatenate([[0], np.cumsum(sq)])
    n = len(audio)
    window_energy = cumsum[target_len:n + 1] - cumsum[:n - target_len + 1]
    best_start = int(np.argmax(window_energy))

    if jitter > 0:
        shift = np.random.randint(-jitter, jitter + 1)
        best_start = max(0, min(best_start + shift, n - target_len))

    return audio[best_start:best_start + target_len].copy()


def _to_mfcc(audio):
    """Extract MFCC from already-cropped audio (exactly target_len samples)."""
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
    return mfcc[..., np.newaxis]


def _augment(audio):
    """Add noise and volume variation to already-cropped audio."""
    audio = audio + np.random.uniform(0.001, 0.01) * np.random.randn(len(audio)).astype(np.float32)
    audio = audio * np.random.uniform(0.7, 1.3)
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


# ── Load dataset ──────────────────────────────────────────────────────────────

print("Loading dataset from", DATASET_DIR)

pos_files = sorted([
    os.path.join(DATASET_DIR, "positive", f)
    for f in os.listdir(os.path.join(DATASET_DIR, "positive"))
    if f.endswith(".wav")
])
neg_files = sorted([
    os.path.join(DATASET_DIR, "negative", f)
    for f in os.listdir(os.path.join(DATASET_DIR, "negative"))
    if f.endswith(".wav")
])

assert pos_files, "No positive .wav files found in dataset/positive/"
assert neg_files, "No negative .wav files found in dataset/negative/"

X, y, groups = [], [], []

print(f"  {len(pos_files)} positive files (+ {AUGMENT_COPIES} augmented each)")
print("  Using energy-based cropping for positives (centers on speech)")
for path in pos_files:
    stem = os.path.splitext(os.path.basename(path))[0]
    group_id = f"pos_{stem}"
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    X.append(_to_mfcc(_energy_crop(audio, TARGET_LENGTH)))
    y.append(1)
    groups.append(group_id)
    for _ in range(AUGMENT_COPIES):
        cropped = _energy_crop(audio, TARGET_LENGTH, jitter=1600)
        X.append(_to_mfcc(_augment(cropped)))
        y.append(1)
        groups.append(group_id)

print(f"  {len(neg_files)} negative files (+ {AUGMENT_COPIES} augmented each)")
print("  Using random cropping for negatives")
for path in neg_files:
    stem = os.path.splitext(os.path.basename(path))[0]
    group_id = f"neg_{stem}"
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)

    X.append(_to_mfcc(_random_crop(audio, TARGET_LENGTH)))
    y.append(0)
    groups.append(group_id)
    for _ in range(AUGMENT_COPIES):
        cropped = _random_crop(audio, TARGET_LENGTH)
        X.append(_to_mfcc(_augment(cropped)))
        y.append(0)
        groups.append(group_id)

X      = np.array(X,      dtype=np.float32)
y      = np.array(y,      dtype=np.int32)
groups = np.array(groups,  dtype=object)

pos_count = int(np.sum(y == 1))
neg_count = int(np.sum(y == 0))
print(f"\nDataset: {pos_count} pos + {neg_count} neg = {len(y)} total")
print(f"  Unique groups: {len(np.unique(groups))}")
print(f"  Feature shape: {X[0].shape}")


# ── Group-aware split ─────────────────────────────────────────────────────────

gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss_outer.split(X, y, groups))

gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx_rel, test_idx_rel = next(gss_inner.split(
    X[temp_idx], y[temp_idx], groups[temp_idx]))
val_idx  = temp_idx[val_idx_rel]
test_idx = temp_idx[test_idx_rel]

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print(f"\nSplits:")
print(f"  Train: {len(y_train)} (pos={int(np.sum(y_train==1))}, neg={int(np.sum(y_train==0))})")
print(f"  Val:   {len(y_val)} (pos={int(np.sum(y_val==1))}, neg={int(np.sum(y_val==0))})")
print(f"  Test:  {len(y_test)} (pos={int(np.sum(y_test==1))}, neg={int(np.sum(y_test==0))})")


# ── Normalize ─────────────────────────────────────────────────────────────────

mean = float(X_train.mean())
std  = float(X_train.std())
print(f"\nNormalization: MFCC_MEAN={mean:.8f}, MFCC_STD={std:.8f}")

X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
X_test  = (X_test  - mean) / std

y_train_oh = tf.keras.utils.to_categorical(y_train, 2)
y_val_oh   = tf.keras.utils.to_categorical(y_val,   2)
y_test_oh  = tf.keras.utils.to_categorical(y_test,  2)


# ── Model ─────────────────────────────────────────────────────────────────────

def build_model():
    inp = tf.keras.Input(shape=(N_MFCC, N_FRAMES, 1))
    x = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inp)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    out = tf.keras.layers.Dense(2, activation='softmax')(x)
    model = tf.keras.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    return model

model = build_model()
model.summary()


# ── Callbacks ─────────────────────────────────────────────────────────────────

model_path = os.path.join(OUTPUT_DIR, "voice_gate.keras")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor='val_loss', save_best_only=True, verbose=0
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1
    ),
]


# ── Train ─────────────────────────────────────────────────────────────────────

print("\nTraining (max %d epochs, early-stopping on val_loss)..." % EPOCHS)
history = model.fit(
    X_train, y_train_oh,
    validation_data=(X_val, y_val_oh),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# ── Evaluate ──────────────────────────────────────────────────────────────────

loss, acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\nTest accuracy: {acc*100:.2f}%  (loss: {loss:.4f})")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion matrix:")
print(f"  TP={tp}  FN={fn}")
print(f"  FP={fp}  TN={tn}")

confidences = np.max(y_pred_probs, axis=1)
print(f"\nCalibration check:")
print(f"  Mean confidence:   {confidences.mean():.3f}")
print(f"  Predictions >99%:  {int(np.sum(confidences > 0.99))} / {len(confidences)}")
print(f"  Predictions >95%:  {int(np.sum(confidences > 0.95))} / {len(confidences)}")
print(f"  Predictions <80%:  {int(np.sum(confidences < 0.80))} / {len(confidences)}")


# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'],     label='train')
axes[0].plot(history.history['val_accuracy'], label='val')
axes[0].set_title('Accuracy'); axes[0].legend()
axes[1].plot(history.history['loss'],         label='train')
axes[1].plot(history.history['val_loss'],     label='val')
axes[1].set_title('Loss'); axes[1].legend()
plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "training_plot.png")
plt.savefig(plot_path)
print(f"\nPlot saved: {plot_path}")


# ── Save normalization ────────────────────────────────────────────────────────

norm_path = os.path.join(OUTPUT_DIR, "normalization.txt")
with open(norm_path, "w") as f:
    f.write(f"{mean:.10f}\n{std:.10f}\n")

print(f"\nKeras model: {model_path}")
print(f"Normalization: {norm_path}")
print("Run export_weights.py next to generate model_data.h / .c")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Test accuracy : {acc*100:.2f}%")
print(f"MFCC_MEAN     : {mean:.8f}f")
print(f"MFCC_STD      : {std:.8f}f")
print(f"Epochs run    : {len(history.history['loss'])}")
print("=" * 60)
