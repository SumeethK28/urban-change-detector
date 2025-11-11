"""
Project configuration file for Urban Change Detection
"""

import os
from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset directories
DATA_DIR = BASE_DIR / "data"
TRAIN_A_DIR = DATA_DIR / "train/A"
TRAIN_B_DIR = DATA_DIR / "train/B"
LABEL_DIR = DATA_DIR / "train/label"

# Image parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 4
EPOCHS = 25

# Model saving path
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "unet_change_detector.h5"

os.makedirs(MODEL_DIR, exist_ok=True)
