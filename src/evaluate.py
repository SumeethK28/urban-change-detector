import tensorflow as tf
from sklearn.metrics import f1_score
import numpy as np
from data_loader import load_dataset
from config import MODEL_SAVE_PATH

model = tf.keras.models.load_model(MODEL_SAVE_PATH)

X1_test, X2_test, y_test = load_dataset("test")

preds = model.predict([X1_test, X2_test])
preds = (preds > 0.5).astype(np.uint8)

# Convert ground-truth to binary labels to match preds
y_true = (y_test > 0.5).astype(np.uint8)

f1 = f1_score(y_true.flatten(), preds.flatten(), average='binary', zero_division=0)
print(f"F1 Score: {f1:.4f}")
