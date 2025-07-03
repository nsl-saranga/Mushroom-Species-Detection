import os;
import cv2
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

directory_path = 'Mushrooms'
image_paths = []
labels = []

for category_name in os.listdir(directory_path):
    category_path = os.path.join(directory_path, category_name)

    if os.path.isdir(category_path):
        for image_name in os.listdir(category_path):
            if image_name.lower().endswith(("jpg", "jpeg","png")):
                image_path = os.path.join(category_path, image_name)
                image_paths.append(image_path)
                labels.append(category_name)


counts = Counter(labels)
class_names = list(counts.keys())
frequencies = list(counts.values())

sorted_indices = np.argsort(frequencies)[::-1]
class_names = [class_names[i] for i in sorted_indices]
frequencies = [frequencies[i] for i in sorted_indices]

plt.figure(figsize=(10, 5))
plt.bar(class_names, frequencies, color='skyblue')
plt.title("Overall Dataset Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


index = 500  # Change this to the index you want

if 0 <= index < len(image_paths):
    selected_image_path = image_paths[index]
    print(f"Loading image: {selected_image_path}")

    img = cv2.imread(selected_image_path)

    # Display the image
    cv2.imshow("Selected Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Invalid index. Please choose a value between 0 and", len(image_paths) - 1)
