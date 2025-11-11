"""
Loads LEVIR-CD dataset and prepares TensorFlow dataset.
Each sample: (before_image, after_image, label_mask)
"""

import tensorflow as tf
import os
import numpy as np
import cv2
from config import IMG_HEIGHT, IMG_WIDTH, TRAIN_A_DIR, TRAIN_B_DIR, LABEL_DIR

def load_image_pair(filename):
    before_path = os.path.join(TRAIN_A_DIR, filename)
    after_path = os.path.join(TRAIN_B_DIR, filename)
    label_path = os.path.join(LABEL_DIR, filename)

    before = cv2.imread(before_path)
    after = cv2.imread(after_path)
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

    before = cv2.resize(before, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
    after = cv2.resize(after, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
    label = cv2.resize(label, (IMG_WIDTH, IMG_HEIGHT)) / 255.0

    return before, after, np.expand_dims(label, axis=-1)

def build_tf_dataset():
    filenames = os.listdir(TRAIN_A_DIR)
    before_list, after_list, label_list = [], [], []

    for f in filenames[:200]:  # limit for testing
        b, a, l = load_image_pair(f)
        before_list.append(b)
        after_list.append(a)
        label_list.append(l)

    before_arr = np.array(before_list, dtype=np.float32)
    after_arr = np.array(after_list, dtype=np.float32)
    label_arr = np.array(label_list, dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(((before_arr, after_arr), label_arr))
    dataset = dataset.shuffle(100).batch(4).prefetch(tf.data.AUTOTUNE)
    return dataset
