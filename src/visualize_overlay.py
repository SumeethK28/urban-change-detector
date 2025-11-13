import cv2
import numpy as np
import tensorflow as tf
from data_loader import load_dataset
from config import MODEL_SAVE_PATH

def overlay_changes(before, after, pred):
    overlay = after.copy()
    mask = (pred.squeeze() > 0.5).astype(np.uint8)
    overlay[mask == 1] = [255, 0, 0]  # red overlay for changed areas
    blended = cv2.addWeighted(after, 0.7, overlay, 0.3, 0)
    return blended

if __name__ == "__main__":
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    X1, X2, y = load_dataset("test")
    preds = model.predict([X1, X2])

    idx = 5
    before = (X1[idx] * 255).astype(np.uint8)
    after = (X2[idx] * 255).astype(np.uint8)
    pred = preds[idx]

    result = overlay_changes(before, after, pred)
    cv2.imwrite("output_overlay.png", result)
    print("✅ Saved overlay as output_overlay.png")
