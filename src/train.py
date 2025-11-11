import tensorflow as tf
from model import build_siamese_unet
from data_loader import build_tf_dataset
from config import MODEL_PATH

def train():
    print("Loading dataset (tiny test)...")
    dataset = build_tf_dataset()

    print("Building model...")
    model = build_siamese_unet()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Running quick 1-epoch test...")
    model.fit(dataset.take(5), epochs=1)   # just 5 batches

    print("✅ Model runs fine, saving test weights...")
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train()
