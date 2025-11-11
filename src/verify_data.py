import cv2
import matplotlib.pyplot as plt
import os
from config import TRAIN_A_DIR, TRAIN_B_DIR, LABEL_DIR

def visualize_sample(index=0):
    files = sorted(os.listdir(TRAIN_A_DIR))
    filename = files[index]

    before = cv2.imread(os.path.join(TRAIN_A_DIR, filename))
    after = cv2.imread(os.path.join(TRAIN_B_DIR, filename))
    label = cv2.imread(os.path.join(LABEL_DIR, filename), cv2.IMREAD_GRAYSCALE)

    before = cv2.cvtColor(before, cv2.COLOR_BGR2RGB)
    after = cv2.cvtColor(after, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(before)
    axes[0].set_title("Before Image")

    axes[1].imshow(after)
    axes[1].set_title("After Image")

    axes[2].imshow(label, cmap='gray')
    axes[2].set_title("Change Mask")

    for ax in axes: ax.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_sample(5)
