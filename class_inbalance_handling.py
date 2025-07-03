from sklearn.utils import class_weight
import numpy as np

def handle_class_imbalance(y_train):
    """
    Computes class weights to handle class imbalance in training data.

    Args:
        y_train (array-like): Numeric class labels for training data (not one-hot encoded)

    Returns:
        dict: Class weights dictionary usable in model.fit(class_weight=...)
    """
    class_weights = class_weight.compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))
    print("Class Weights:", class_weights)
    return class_weights
