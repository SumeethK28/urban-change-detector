"""
Loads LEVIR-CD dataset and prepares TensorFlow dataset.
Each sample: (before_image, after_image, label_mask)
"""

import tensorflow as tf
import os
import cv2
import numpy as np
from config import IMG_HEIGHT, IMG_WIDTH, DATASET_PATH

def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0
    return img.astype(np.float32)

def load_dataset(folder="train"):
    A_path = os.path.join(DATASET_PATH, folder, "A")
    B_path = os.path.join(DATASET_PATH, folder, "B")
    L_path = os.path.join(DATASET_PATH, folder, "label")

    A_images, B_images, labels = [], [], []

    for file in os.listdir(A_path):
        before = load_image(os.path.join(A_path, file))
        after = load_image(os.path.join(B_path, file))
        label = load_image(os.path.join(L_path, file))
        label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        label = np.expand_dims(label, axis=-1) / 255.0

        A_images.append(before)
        B_images.append(after)
        labels.append(label)

    return np.array(A_images), np.array(B_images), np.array(labels)
