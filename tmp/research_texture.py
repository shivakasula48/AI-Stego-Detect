import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steganography.embed import embed_message
from ai_model.predict import predict_image

def test_texture_sensitivity():
    print("--- Testing Texture Sensitivity for Short Messages ---")
    
    # Create a textured image (high detail)
    textured_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    message = "hi"
    
    print("\n[ALPHA=4] Testing short message 'hi' on RANDOM TEXTURE...")
    stego_text = embed_message(textured_img, message)
    cv2.imwrite("stego_text.png", stego_text, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    label_text, conf_text = predict_image("stego_text.png")
    print(f"Result (ALPHA=4, msg='hi', texture=random): {label_text} ({conf_text:.2f}%)")
    
    # CASE: Long message on texture
    long_msg = "A" * 200
    print("\n[ALPHA=4] Testing long message (200 bytes) on RANDOM TEXTURE...")
    stego_text_long = embed_message(textured_img, long_msg)
    cv2.imwrite("stego_text_long.png", stego_text_long, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    label_long, conf_long = predict_image("stego_text_long.png")
    print(f"Result (ALPHA=4, msg_len=200, texture=random): {label_long} ({conf_long:.2f}%)")

if __name__ == "__main__":
    test_texture_sensitivity()
