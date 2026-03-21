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
EPOCHS         = 100
AUGMENT_COPIES = 4

DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
OUTPUT_DIR  = os.path.join(os.path.dirname(__file__), "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# ── Audio preprocessing ───────────────────────────────────────────────────────
# center=False ensures frame boundaries match C implementation exactly:
# frame i starts at sample i*HOP_LENGTH, covering samples [i*HOP, i*HOP+N_FFT).

def _extract_mfcc(audio):
    if len(audio) < TARGET_LENGTH:
        audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))
    else:
        audio = audio[:TARGET_LENGTH]
    mfcc = librosa.feature.mfcc(
        y=audio, sr=SAMPLE_RATE,
        n_mfcc=N_MFCC, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
        center=False
    )
    # With center=False and 16000 samples: exactly 99 frames — no pad/clip needed.
    # Keep the clip as a safety guard.
    if mfcc.shape[1] < N_FRAMES:
        mfcc = np.pad(mfcc, ((0, 0), (0, N_FRAMES - mfcc.shape[1])))
    else:
        mfcc = mfcc[:, :N_FRAMES]
    return mfcc[..., np.newaxis]   # (13, 99, 1)


def preprocess(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return _extract_mfcc(audio)


def preprocess_augmented(path):
    audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    if len(audio) < TARGET_LENGTH:
        audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)))
    else:
        audio = audio[:TARGET_LENGTH]
    # Noise, volume, time-shift
    noise_level = np.random.uniform(0.001, 0.01)
    audio = audio + noise_level * np.random.randn(len(audio))
    audio = audio * np.random.uniform(0.7, 1.3)
    shift = np.random.randint(-1600, 1600)
    audio = np.roll(audio, shift)
    audio = np.clip(audio, -1.0, 1.0)
    return _extract_mfcc(audio)


# ── Load dataset with group tracking ─────────────────────────────────────────
# Groups prevent the same speaker / recording from appearing in both train and
# test, fixing the data leakage that caused the inflated 99.77% accuracy.

print("Loading dataset...")

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

X, y, groups = [], [], []

print(f"  Processing {len(pos_files)} positive files (+ {AUGMENT_COPIES} augmented copies each)...")
for path in pos_files:
    stem = os.path.splitext(os.path.basename(path))[0]  # e.g. "positive_5"
    group_id = f"pos_{stem}"
    X.append(preprocess(path))
    y.append(1)
    groups.append(group_id)
    for _ in range(AUGMENT_COPIES):
        X.append(preprocess_augmented(path))
        y.append(1)
        groups.append(group_id)  # augmented copies share the same group

print(f"  Processing {len(neg_files)} negative files...")
for path in neg_files:
    # filename: word_SPEAKERHASH_nohash_N.wav  → group on speaker hash
    parts = os.path.splitext(os.path.basename(path))[0].split("_")
    group_id = f"neg_{parts[1]}" if len(parts) >= 2 else f"neg_{parts[0]}"
    X.append(preprocess(path))
    y.append(0)
    groups.append(group_id)

X      = np.array(X,      dtype=np.float32)
y      = np.array(y,      dtype=np.int32)
groups = np.array(groups, dtype=object)

pos_count = int(np.sum(y == 1))
neg_count = int(np.sum(y == 0))
print(f"\nDataset loaded:")
print(f"  Positive samples : {pos_count}")
print(f"  Negative samples : {neg_count}")
print(f"  Total            : {len(y)}")
print(f"  Unique groups    : {len(np.unique(groups))}")
print(f"  Feature shape    : {X[0].shape}")


# ── Group-aware split ─────────────────────────────────────────────────────────
# GroupShuffleSplit ensures no recording appears in both train and test.

gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(gss_outer.split(X, y, groups))

gss_inner = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx_rel, test_idx_rel = next(gss_inner.split(X[temp_idx], y[temp_idx], groups[temp_idx]))
val_idx  = temp_idx[val_idx_rel]
test_idx = temp_idx[test_idx_rel]

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print(f"\nGroup-aware splits:")
print(f"  Train : {len(y_train)}  (pos={int(np.sum(y_train==1))}, neg={int(np.sum(y_train==0))})")
print(f"  Val   : {len(y_val)}   (pos={int(np.sum(y_val==1))},  neg={int(np.sum(y_val==0))})")
print(f"  Test  : {len(y_test)}   (pos={int(np.sum(y_test==1))},  neg={int(np.sum(y_test==0))})")


# ── Normalize ─────────────────────────────────────────────────────────────────
mean = float(X_train.mean())
std  = float(X_train.std())
print(f"\n*** Normalization constants (copy these into model_data.h) ***")
print(f"    MFCC_MEAN = {mean:.8f}f")
print(f"    MFCC_STD  = {std:.8f}f")

X_train = (X_train - mean) / std
X_val   = (X_val   - mean) / std
X_test  = (X_test  - mean) / std


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
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
model.summary()


# ── Callbacks ─────────────────────────────────────────────────────────────────
model_path = os.path.join(OUTPUT_DIR, "voice_gate.keras")
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        model_path, monitor='val_accuracy', save_best_only=True, verbose=0
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5, patience=5, verbose=1
    ),
]


# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# ── Evaluate ──────────────────────────────────────────────────────────────────
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest accuracy: {acc*100:.2f}%")

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion matrix:")
print(f"  True Positives  : {tp}")
print(f"  True Negatives  : {tn}")
print(f"  False Positives : {fp}")
print(f"  False Negatives : {fn}")

high_conf = int(np.sum(y_pred_probs[:, 1] > 0.9))
print(f"\nPositive predictions above 0.9 confidence: {high_conf} / {int(np.sum(y_test==1))}")


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
print(f"\nTraining plot saved: {plot_path}")


# ── Save normalization for export script ──────────────────────────────────────
norm_path = os.path.join(OUTPUT_DIR, "normalization.txt")
with open(norm_path, "w") as f:
    f.write(f"{mean:.10f}\n{std:.10f}\n")
print(f"Normalization saved: {norm_path}")

print(f"\nKeras model saved: {model_path}")
print("\nRun export_weights.py next to generate model_data.h and model_data.c")

print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
print(f"Test accuracy  : {acc*100:.2f}%")
print(f"MFCC_MEAN      : {mean:.8f}f")
print(f"MFCC_STD       : {std:.8f}f")
print(f"Input shape    : {N_MFCC} x {N_FRAMES} = {N_MFCC*N_FRAMES} features")
print("="*60)
