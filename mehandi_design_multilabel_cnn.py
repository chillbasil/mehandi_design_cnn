"""
mehandi_design_multilabel_cnn.py
------------------------------------
Train, evaluate, and predict multi-label Mehandi Designs
using a CSV-based Roboflow dataset.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

DATA_DIR = "."
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
CLASSES = ["flower", "leaf", "lotus", "mango", "peacock"]

def load_dataset(csv_path, images_dir):
    """Load images and labels from a Roboflow CSV (multi-label format)."""
    df = pd.read_csv(csv_path)
    images, labels = [], []

    for _, row in df.iterrows():
        img_path = os.path.join(images_dir, row["filename"])
        if not os.path.exists(img_path):
            continue  # skip missing files

        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(row[CLASSES].values.astype("float32"))

    return np.array(images), np.array(labels)

def create_dataset(split):
    """Load one dataset split (train, valid, or test)."""
    folder = os.path.join(DATA_DIR, split)
    csv_path = os.path.join(folder, "_classes.csv")
    images_dir = folder

    X, y = load_dataset(csv_path, images_dir)
    print(f"Loaded {split}: {X.shape[0]} images, {y.shape[1]} labels")
    return X, y

def predict_single_image(model, image_path):
    """Load and predict a single new image."""
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    threshold = 0.5
    predicted_labels = [CLASSES[i] for i, p in enumerate(preds) if p >= threshold]

    print("\nPrediction Results:")
    for i, label in enumerate(CLASSES):
        print(f"{label:10s}: {preds[i]:.3f} {'✔' if preds[i] >= threshold else ''}")

    if not predicted_labels:
        predicted_labels = ["(no confident label detected)"]

    plt.imshow(load_img(image_path))
    plt.title(f"Predicted: {', '.join(predicted_labels)}")
    plt.axis("off")
    plt.show()

print("Loading dataset...")
X_train, y_train = create_dataset("train")
X_valid, y_valid = create_dataset("valid")
X_test, y_test = create_dataset("test")

print("Building CNN model...")

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SIZE + (3,)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(CLASSES), activation='sigmoid')  # multi-label output
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

print("Evaluating model on test set...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.show()

print("Showing sample predictions...")
num_samples = 9
indices = np.random.choice(len(X_test), num_samples, replace=False)
X_sample = X_test[indices]
y_true = y_test[indices]
y_pred = (model.predict(X_sample) > 0.5).astype(int)

plt.figure(figsize=(10, 10))
for i in range(num_samples):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(X_sample[i])
    true_labels = [CLASSES[j] for j in range(len(CLASSES)) if y_true[i][j] == 1]
    pred_labels = [CLASSES[j] for j in range(len(CLASSES)) if y_pred[i][j] == 1]
    plt.title(
        f"T: {','.join(true_labels)}\nP: {','.join(pred_labels)}",
        fontsize=8,
        color="green" if true_labels == pred_labels else "red"
    )
    plt.axis("off")
plt.show()

print("Saving model as 'mehandi_multilabel_cnn.keras'")
model.save("mehandi_multilabel_cnn.keras")
print("Model saved successfully.")

# PREDICT SINGLE IMAGE 

# Uncomment the line below and replace 'sample.jpg' with your image path
# predict_single_image(model, "sample.jpg")
