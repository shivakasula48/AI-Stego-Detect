import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steganography.embed import embed_message
from ai_model.predict import predict_image

def final_verify_ui_logic():
    print("--- Final UI-Logic Verification (Fix-Detection) ---")
    
    # 1. Simulate Upload (Large Image)
    large_img = np.random.randint(50, 200, (640, 480, 3), dtype=np.uint8)
    cv2.imwrite("uploaded_large.png", large_img)
    
    # 2. Simulate app.py's new logic: Resize then Embed
    print("Resizing to 128x128 before embedding...")
    input_img = cv2.imread("uploaded_large.png")
    resized_img = cv2.resize(input_img, (128, 128))
    
    message = "Fixing the detection mismatch for good."
    stego_img = embed_message(resized_img, message)
    cv2.imwrite("stego_fixed_ui.png", stego_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # 3. Predict result
    print("\nPredicting result of stego_fixed_ui.png...")
    label, confidence = predict_image("stego_fixed_ui.png")
    
    print(f"\nResult: {label} ({confidence:.2f}%)")
    
    if label == "Stego Image":
        print("\n✅ SUCCESS: The CNN correctly identified the stego image from the fixed UI logic.")
    else:
        print("\n❌ FAILURE: Check ALPHA or Q values in embed.py.")

if __name__ == "__main__":
    final_verify_ui_logic()
