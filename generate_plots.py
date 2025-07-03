import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def plot_class_distribution(y, label_map=None, title="Class Distribution"):
    counts = Counter(y)
    classes = list(counts.keys())
    frequencies = list(counts.values())

    # If label_map (e.g., {'Shiitake': 0, 'Oyster': 1, ...}) is provided, reverse it
    if label_map:
        class_names = [list(label_map.keys())[list(label_map.values()).index(c)] for c in classes]
    else:
        class_names = [str(c) for c in classes]

    # Sort by frequency (optional)
    sorted_indices = np.argsort(frequencies)[::-1]
    class_names = [class_names[i] for i in sorted_indices]
    frequencies = [frequencies[i] for i in sorted_indices]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(class_names, frequencies, color='skyblue')
    plt.title(title)
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
