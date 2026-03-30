import os
import sys
import numpy as np
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset.generate_stego import generate_stego_dataset
from ai_model.predict import predict_image

def verify_dataset_detection():
    print("--- Verifying Detection on Freshly Generated Dataset ---")
    
    # 1. Generate a small dataset
    print("Generating dataset samples...")
    # This will use the current embed_message
    train_clean, train_stego, val_clean, val_stego = generate_stego_dataset(
        base_dir="tmp/dataset_verify", 
        num_images=4, 
        size=(128, 128)
    )
    
    # 2. Pick one stego image from val/stego
    stego_files = [f for f in os.listdir(val_stego) if f.endswith('.png')]
    if not stego_files:
        print("No stego files generated.")
        return
        
    stego_path = os.path.join(val_stego, stego_files[0])
    print(f"Testing freshly generated stego image: {stego_path}")
    
    # 3. Predict
    label, conf = predict_image(stego_path)
    print(f"\nResult: {label} ({conf:.2f}%)")
    
    if label == "Stego Image":
        print("\n✅ SUCCESS: The CNN detects images from its own training logic.")
    else:
        print("\n❌ FAILURE: Even freshly generated images are NOT detected. This means the model is disconnected from the current code parameters.")

if __name__ == "__main__":
    verify_dataset_detection()
