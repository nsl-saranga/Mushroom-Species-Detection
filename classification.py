import os;
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from model_builder import MushroomClassifier
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

directory_path = 'Mushrooms'
image_size = (128,128)

X = []
y = []

label_map = {}
label_counter = 0

for category_name in os.listdir(directory_path):
    category_path = os.path.join(directory_path, category_name)

    if os.path.isdir(category_path):
        if category_name not in label_map:
            label_map[category_name] = label_counter
            label_counter += 1
        label = label_map[category_name]

        for image_name in os.listdir(category_path):
            if image_name.lower().endswith(("jpg", "jpeg","png")):
                image_path = os.path.join(category_path, image_name)
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Skipping corrupted image: {image_path}")
                    continue

                image = cv2.resize(image, image_size)
                X.append(image)
                y.append(label)

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)
print(f"Loaded {len(X)} images across {len(label_map)} categories.")

num_classes = len(label_map)

# scaling
X = X/255.0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

model = MushroomClassifier(num_classes = num_classes)
model.summary()

# Train
history = model.train(X_train, y_train_cat, epochs=10, batch_size=32, validation_split=0.1)

# Evaluate
loss, acc = model.evaluate(X_test, y_test_cat)

print(f"Test Accuracy: {acc * 100:.2f}%")

import matplotlib.pyplot as plt

# Plot training & validation accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


