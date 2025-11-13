import os

DATASET_PATH = os.path.join("data", "LEVIR_CD")
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 8
EPOCHS = 30
LR = 1e-4
MODEL_SAVE_PATH = "models/siamese_unet_tf.keras"
