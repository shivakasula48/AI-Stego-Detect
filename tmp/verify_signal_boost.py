import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steganography.embed import embed_message
from ai_model.predict import predict_image

def verify_signal_boost():
    print("--- Verifying if ALPHA=16 overcomes Texture Masking ---")
    
    # Create a textured image
    np.random.seed(42) # Consistent texture
    textured_img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    message = "hi"
    
    import steganography.embed
    
    # CASE: ALPHA=16 on Texture
    steganography.embed.ALPHA = 16
    print("\n[ALPHA=16] Testing short message 'hi' on RANDOM TEXTURE...")
    stego_16 = embed_message(textured_img, message)
    cv2.imwrite("stego_16_text.png", stego_16, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    label_16, conf_16 = predict_image("stego_16_text.png")
    print(f"Result (ALPHA=16, msg='hi', texture=random): {label_16} ({conf_16:.2f}%)")
    
    # CASE: ALPHA=32 on Texture (if 16 isn't enough)
    steganography.embed.ALPHA = 32
    print("\n[ALPHA=32] Testing short message 'hi' on RANDOM TEXTURE...")
    stego_32 = embed_message(textured_img, message)
    cv2.imwrite("stego_32_text.png", stego_32, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    label_32, conf_32 = predict_image("stego_32_text.png")
    print(f"Result (ALPHA=32, msg='hi', texture=random): {label_32} ({conf_32:.2f}%)")

if __name__ == "__main__":
    verify_signal_boost()
