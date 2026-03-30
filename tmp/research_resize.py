import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steganography.embed import embed_message
from ai_model.predict import predict_image

def research_resize():
    print("--- Researching Resize Impact on Detection ---")
    
    # 1. Create a large "original" image (e.g. 512x512)
    large_img = np.random.randint(50, 200, (512, 512, 3), dtype=np.uint8)
    message = "Resizing test message"
    
    # CASE A: Embed into Large, then Detection resizes it
    print("\n[Case A] Embedding into 512x512, then Detection resizes to 128x128:")
    stego_large = embed_message(large_img, message)
    cv2.imwrite("stego_large.png", stego_large, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    label_a, conf_a = predict_image("stego_large.png")
    print(f"Result A: {label_a} ({conf_a:.2f}%)")
    
    # CASE B: Resize to 128x128 FIRST, then Embed
    print("\n[Case B] Resizing to 128x128 FIRST, then Embedding:")
    small_img = cv2.resize(large_img, (128, 128))
    stego_small = embed_message(small_img, message)
    cv2.imwrite("stego_small.png", stego_small, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    label_b, conf_b = predict_image("stego_small.png")
    print(f"Result B: {label_b} ({conf_b:.2f}%)")
    
    if label_a == "Clean Image" and label_b == "Stego Image":
        print("\n✅ HYPOTHESIS CONFIRMED: Resizing after embedding destroys the signal. We must resize BEFORE embedding for this CNN.")
    else:
        print("\n❌ HYPOTHESIS NOT FULLY CONFIRMED. Check other factors.")

if __name__ == "__main__":
    research_resize()
