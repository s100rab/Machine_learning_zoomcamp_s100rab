import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_images(data_dir, img_size=(64, 64)):
    """
    Load images from a directory and preprocess them.
    
    Args:
    - data_dir (str): Path to the dataset directory.
    - img_size (tuple): Desired image size (width, height).
    
    Returns:
    - images (np.array): Array of preprocessed images.
    """
    images = []
    labels = []

    for label, folder in enumerate(os.listdir(data_dir)):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                img = cv2.resize(img, img_size)  # Resize images
                images.append(img)
                labels.append(label)

    images = np.array(images, dtype='float32')
    images = images / 255.0  # Normalize to [0, 1]
    labels = np.array(labels)

    return images, labels

def prepare_data(data_dir):
    """
    Prepare training and testing data.
    
    Args:
    - data_dir (str): Path to the dataset directory.
    
    Returns:
    - X_train, X_test, y_train, y_test: Train/test split of data.
    """
    images, labels = load_images(data_dir)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension
    X_test = np.expand_dims(X_test, axis=-1)

    return X_train, X_test, y_train, y_test
