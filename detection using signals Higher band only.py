
# DRONE RF SIGNAL CLASSIFIER — CNN + STFT
# HIGH BAND VERSION — only {CODE}H_{i}.csv files
# GOOGLE COLAB VERSION
# Device : USRP-2943
# fs : 200 MHz | Band : 40 MHz
# Files : 227 H files in Google Drive

#CELL 1: Check GPU ────────────────────────────────────────
import tensorflow as tf
print("TensorFlow version :", tf.__version__)
print("GPU available :", tf.config.list_physical_devices('GPU'))
# Runtime → Change runtime type → T4 GPU → Save


# CELL 2: Mount Google Drive ───────────────────────────────
from google.colab import drive
drive.mount('/content/drive')


# CELL 3: Imports ──────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
from collections import Counter
from scipy.signal import stft
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import psutil
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')


# CELL 4: CONFIGURATION ────────────────────────────────────
DATASET_FOLDER = "/content/drive/MyDrive/ALL DATA 4"
CACHE_FOLDER = "/content/stft_cache"
os.makedirs(CACHE_FOLDER, exist_ok=True)

CLASS_MAP = {
    "00000": {"label": 0, "name": "No Drone (Background)", "code": "00000", "n_files": 11},
    "10000": {"label": 1, "name": "Bebop - ON/Connected", "code": "10000", "n_files": 6},
    "10001": {"label": 2, "name": "Bebop - Hovering", "code": "10001", "n_files": 6},
    "10010": {"label": 3, "name": "Bebop - Flying (no rec)", "code": "10010", "n_files": 6},
    "10011": {"label": 4, "name": "Bebop - Flying (rec)", "code": "10011", "n_files": 6},
    "10100": {"label": 5, "name": "AR Drone - ON/Connected", "code": "10100", "n_files": 6},
    "10101": {"label": 6, "name": "AR Drone - Hovering", "code": "10101", "n_files": 6},
    "10110": {"label": 7, "name": "AR Drone - Flying (no rec)", "code": "10110", "n_files": 6},
    "10111": {"label": 8, "name": "AR Drone - Flying (rec)", "code": "10111", "n_files": 6},
    "11000": {"label": 9, "name": "Phantom - ON/Connected", "code": "11000", "n_files": 6},
}

# Hardware (USRP-2943)
FS = 200_000_000
BAND = 40_000_000

# STFT parameters
NPERSEG = 256
NOVERLAP = 128

# Windowing
WINDOW_SIZE = 2560
WINDOW_STEP = 1280
MAX_WINDOWS = 130# 50 for Colab (12GB RAM)

# Training
N_CLASSES = 10
BATCH_SIZE = 32
EPOCHS = 150
TEST_SIZE = 0.2
VAL_SIZE = 0.1


#CELL 5: LOAD SIGNAL ──────────────────────────────────────
def load_signal(filepath, max_samples=5_000_000):
    """
    Read H CSV file — comma-separated samples per line.
    Applies per-signal normalization.
    """
    values = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if len(values) >= max_samples:
                    break
                for num in line.strip().split(','):
                    num = num.strip()
                    if num:
                        try:
                            values.append(float(num))
                        except ValueError:
                            continue
                        if len(values) >= max_samples:
                            break
    except Exception as e:
        print(f" ⚠️ Error: {e}")
        return np.array([0.0], dtype=np.float32)

    signal = np.array(values, dtype=np.float32)
    std = signal.std()
    if std > 1e-8:
        signal = (signal - signal.mean()) / std
    return signal


#CELL 6: COMPUTE STFT ─────────────────────────────────────
def compute_stft(signal):
    """
    Log scale STFT → amplifies small differences.
    Returns normalized float32 image (257 × 19).
    """
    _, _, Zxx = stft(signal, fs=FS, nperseg=NPERSEG, noverlap=NOVERLAP)
    magnitude = np.abs(Zxx)
    magnitude = np.log1p(magnitude)
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    return magnitude.astype(np.float32)


# CELL 7: EXTRACT WINDOWS ──────────────────────────────────
def extract_windows(signal):
    windows, start = [], 0
    while start + WINDOW_SIZE <= len(signal) and len(windows) < MAX_WINDOWS:
        windows.append(signal[start : start + WINDOW_SIZE])
        start += WINDOW_STEP
    return windows


# CELL 8: BUILD DATASET ────────────────────────────────────
def build_dataset():
    """
    Loads only H files: {CODE}H_{i}.csv
    """
    labels, npy_paths, stft_shapes = [], [], []
    sample_idx = 0

    print("\n" + "="*65)
    print(" 📡 DATASET BUILDER — HIGH BAND ONLY (H files)")
    print(" 📍 Platform : Google Colab")
    print("="*65)
    print(f" Reading only : {{CODE}}H_{{i}}.csv files")
    print(f" fs={FS/1e6:.0f}MHz NPERSEG={NPERSEG} "
          f"Window={WINDOW_SIZE} MaxWin={MAX_WINDOWS}")
    print("="*65)
    print(f"\n {'Class':<35} {'Code':<8} {'Files':>6} {'Windows':>10} RAM%")
    print(f" {'-'*65}")

    for code, info in CLASS_MAP.items():
        class_windows = 0

        for i in range(info["n_files"]):

            # ── Only H file ──────────────────────────────────
            high_file = os.path.join(DATASET_FOLDER, f"{code}H_{i}.csv")

            if not os.path.exists(high_file):
                print(f" ⚠️ Missing: {code}H_{i}.csv")
                continue

            signal = load_signal(high_file, max_samples=5_000_000)
            windows = extract_windows(signal)
            del signal
            gc.collect()

            for win in windows:
                stft_img = compute_stft(win)
                npy_path = os.path.join(CACHE_FOLDER, f"s_{sample_idx}.npy")
                np.save(npy_path, stft_img)
                stft_shapes.append(stft_img.shape)
                npy_paths.append(npy_path)
                labels.append(info["label"])
                sample_idx += 1
                class_windows += 1
                del stft_img
            del windows
            gc.collect()

        ram = psutil.virtual_memory()
        print(f" ✅ {info['name']:<35} {code:<8} "
              f"{info['n_files']:>6} {class_windows:>10} {ram.percent:.0f}%")

    if len(npy_paths) == 0:
        raise ValueError(f"❌ No files loaded! Check: {DATASET_FOLDER}")

    target_shape = Counter(stft_shapes).most_common(1)[0][0]
    total = len(npy_paths)

    print(f"\n 📊 STFT shape : {target_shape}")
    print(f" 📊 Total : {total:,} samples")
    print(f" 📊 RAM needed : {total*target_shape[0]*target_shape[1]*4/1e6:.0f} MB")
    print(f"\n Loading into RAM...")

    X = np.zeros((total, target_shape[0], target_shape[1], 1), dtype=np.float32)
    for idx, p in enumerate(npy_paths):
        img = np.load(p)
        if img.shape != target_shape:
            r = np.zeros(target_shape, dtype=np.float32)
            f = min(img.shape[0], target_shape[0])
            t = min(img.shape[1], target_shape[1])
            r[:f, :t] = img[:f, :t]
            X[idx, :, :, 0] = r
        else:
            X[idx, :, :, 0] = img

    y = np.array(labels)
    print(f"\n 📊 Dataset : {X.shape} RAM: {X.nbytes/1e6:.0f} MB")
    print(f"\n 📊 Samples per class:")
    for info in CLASS_MAP.values():
        n = int(np.sum(y == info['label']))
        print(f" {info['code']} {info['name']:<35} {n:>8} "
              f"{'⚠️' if n==0 else '✅'}")
    return X, y


#CELL 9: BUILD CNN ────────────────────────────────────────
def build_cnn(input_shape, n_classes=N_CLASSES):
    """
    3-block CNN with GlobalAveragePooling + L2 regularization.
    Input : (257 × 19 × 1)
    Output : (10,) softmax
    """
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same',
                      input_shape=input_shape,
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(64, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
         layers.Conv2D(128, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((1,1)),
        layers.Dropout(0.25),


        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


#CELL 10: TRAIN ───────────────────────────────────────────
def train(X, y):
    y_cat = to_categorical(y, num_classes=N_CLASSES)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=TEST_SIZE, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, random_state=42)

    print(f"\n 🔀 Train : {len(X_train):,} Val : {len(X_val):,} Test : {len(X_test):,}")

    y_int = np.argmax(y_train, axis=1)
    cw = compute_class_weight('balanced', classes=np.arange(N_CLASSES), y=y_int)
    cw_dict = dict(enumerate(cw))
    print(f"\n ⚖️ Class weights: { {k: round(v,2) for k,v in cw_dict.items()} }")

    model = build_cnn(X_train.shape[1:])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=99999,
                      restore_best_weights=True,
                      min_delta=1e-7, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=20, min_lr=1e-6, verbose=1)
    ]

    print("\n🚀 Training...\n")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw_dict,
        callbacks=callbacks
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n{'='*50}")
    print(f" ✅ Test Accuracy : {acc*100:.2f}%")
    print(f" ✅ Test Loss : {loss:.4f}")
    print(f"{'='*50}")
    return model, history, X_test, y_test


#CELL 11: PLOT HISTORY ────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training History — High Band (H files)',
                 fontsize=15, fontweight='bold')
    axes[0].plot(history.history['accuracy'], label='Train', color='royalblue')
    axes[0].plot(history.history['val_accuracy'], label='Validation', color='tomato')
    axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch')
    axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[1].plot(history.history['loss'], label='Train', color='royalblue')
    axes[1].plot(history.history['val_loss'], label='Validation', color='tomato')
    axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch')
    axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('/content/training_history_H.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📸 Saved: training_history_H.png")


#CELL 12: CONFUSION MATRIX ────────────────────────────────
def plot_confusion_matrix(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    codes = [v['code'] for v in CLASS_MAP.values()]
    names = [v['name'] for v in CLASS_MAP.values()]

    plt.figure(figsize=(13, 11))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=codes, yticklabels=codes,
                linewidths=0.5, linecolor='gray')
    plt.title('Confusion Matrix — High Band (H files)',
              fontsize=15, fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig('/content/confusion_matrix_H.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📸 Saved: confusion_matrix_H.png")

    accs = [np.mean(y_pred[y_true==c]==c)*100
              if (y_true==c).sum()>0 else 0 for c in range(N_CLASSES)]
    colors = ['#4CAF50' if a>=80 else '#FF9800' if a>=60 else '#F44336' for a in accs]
    plt.figure(figsize=(14, 5))
    bars = plt.bar(range(N_CLASSES), accs, color=colors, edgecolor='white')
    plt.xticks(range(N_CLASSES), codes, rotation=45, ha='right', fontsize=9)
    plt.ylabel('Accuracy (%)'); plt.ylim(0, 115)
    plt.title('Per-Class Accuracy — High Band', fontsize=14, fontweight='bold')
    plt.axhline(y=80, color='green', linestyle='--', alpha=0.5, label='80%')
    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.4)
    for bar, a in zip(bars, accs):
        plt.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                 f'{a:.0f}%', ha='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/content/per_class_accuracy_H.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📸 Saved: per_class_accuracy_H.png")

    print("\n📋 Classification Report:\n")
    print(classification_report(y_true, y_pred, target_names=names))


#CELL 13: PREDICT SINGLE FILE ─────────────────────────────
def predict_single(model, high_path, target_shape):
    """
    Predict from a single H file only.

    Usage:
        predict_single(
            model,
            high_path = "/content/drive/MyDrive/All data/4/10101H_0.csv",
            target_shape = X.shape[1:3]
        )
    """
    signal = load_signal(high_path, max_samples=5_000_000)
    window = signal[:WINDOW_SIZE] if len(signal) >= WINDOW_SIZE else signal
    stft_img = compute_stft(window)
    resized = np.zeros(target_shape, dtype=np.float32)
    f = min(stft_img.shape[0], target_shape[0])
    t = min(stft_img.shape[1], target_shape[1])
    resized[:f, :t] = stft_img[:f, :t]
    probs = model.predict(resized[np.newaxis,...,np.newaxis], verbose=0)[0]
    pred = np.argmax(probs)

    print(f"\n{'='*62}")
    print(f" 🎯 PREDICTION RESULT — High Band Only")
    print(f"{'='*62}")
    for info in CLASS_MAP.values():
        if info["label"] == pred:
            print(f" Class : {info['name']}")
            print(f" Binary Code: {info['code']}")
            print(f" Confidence : {probs[pred]*100:.1f}%")
    print(f"\n {'Code':<8} {'Name':<35} {'Prob':>6} Bar")
    print(f" {'-'*68}")
    for info in CLASS_MAP.values():
        p = probs[info['label']] * 100
        bar = '█' * int(p/4)
        mk = ' ◀ PREDICTED' if info['label'] == pred else ''
        print(f" {info['code']:<8} {info['name']:<35} {p:5.1f}% {bar}{mk}")
    print(f"{'='*62}\n")


#CELL 14: SAVE / LOAD MODEL ───────────────────────────────
def save_model(model, path="/content/drive/MyDrive/drone_cnn_H.h5"):
    model.save(path)
    print(f"✅ Model saved: {path}")

def load_saved_model(path="/content/drive/MyDrive/drone_cnn_H.h5"):
    model = tf.keras.models.load_model(path)
    print(f"✅ Model loaded: {path}")
    return model


#CELL 15: RUN ALL ─────────────────────────────────────────
X, y = build_dataset()
model, history, X_test, y_test = train(X, y)
plot_history(history)
plot_confusion_matrix(model, X_test, y_test)
save_model(model)

# Predict single H file — uncomment and edit path
# predict_single(
# model,
# high_path = "/content/drive/MyDrive/All data/4/10101H_0.csv",
# target_shape = X.shape[1:3]
# )