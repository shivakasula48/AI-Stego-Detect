import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steganography.embed import embed_message
from ai_model.predict import predict_image

def test_signal_strength():
    print("--- Testing Stego Signal Strength (ALPHA) (Fixing Overflow) ---")
    
    # Create a textured image
    textured_img = np.random.randint(50, 150, (128, 128, 3), dtype=np.uint8)
    # Simple gradient texture
    for i in range(128):
        textured_img[i, :, 0] = (textured_img[i, :, 0] + i) % 256
        
    message = "Testing if higher ALPHA helps detection on textures."
    
    import steganography.embed
    print(f"Current ALPHA in code: {steganography.embed.ALPHA}")
    
    # CASE 1: ALPHA=4 (Current)
    stego_curr = embed_message(textured_img, message)
    cv2.imwrite("stego_curr.png", stego_curr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    label_curr, conf_curr = predict_image("stego_curr.png")
    print(f"Result (ALPHA={steganography.embed.ALPHA}): {label_curr} ({conf_curr:.2f}%)")
    
    # CASE 2: ALPHA=16
    old_alpha = steganography.embed.ALPHA
    steganography.embed.ALPHA = 16
    stego_high = embed_message(textured_img, message)
    cv2.imwrite("stego_high.png", stego_high, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    label_high, conf_high = predict_image("stego_high.png")
    print(f"Result (ALPHA=16): {label_high} ({conf_high:.2f}%)")
    
    steganography.embed.ALPHA = old_alpha

if __name__ == "__main__":
    test_signal_strength()
