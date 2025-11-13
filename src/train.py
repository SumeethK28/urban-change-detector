from model import SiameseUNet
from data_loader import load_dataset
from config import EPOCHS, LR, MODEL_SAVE_PATH
import tensorflow as tf

if __name__ == "__main__":
    X1_train, X2_train, y_train = load_dataset("train")
    X1_val, X2_val, y_val = load_dataset("val")

    model = SiameseUNet()
    model.compile(optimizer=tf.keras.optimizers.Adam(LR),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    model.fit([X1_train, X2_train], y_train,
              validation_data=([X1_val, X2_val], y_val),
              epochs=EPOCHS,
              batch_size=4)

    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved at {MODEL_SAVE_PATH}")
