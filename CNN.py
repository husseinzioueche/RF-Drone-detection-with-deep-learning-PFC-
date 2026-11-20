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


# ── CELL 13: PREDICT SINGLE FILE ─────────────────────────────
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


# ── CELL 14: SAVE / LOAD MODEL ───────────────────────────────
def save_model(model, path="/content/drive/MyDrive/drone_cnn_H.h5"):
    model.save(path)
    print(f"✅ Model saved: {path}")

def load_saved_model(path="/content/drive/MyDrive/drone_cnn_H.h5"):
    model = tf.keras.models.load_model(path)
    print(f"✅ Model loaded: {path}")
    return model


# ── CELL 15: RUN ALL ─────────────────────────────────────────
X, y = build_dataset()
model, history, X_test, y_test = train(X, y)
plot_history(history)
plot_confusion_matrix(model, X_test, y_test)
save_model(model)