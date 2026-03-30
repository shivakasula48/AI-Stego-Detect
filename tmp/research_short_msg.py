import cv2
import numpy as np
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steganography.embed import embed_message
from ai_model.predict import predict_image

def test_short_message_detection():
    print("--- Testing Short Message Detection on Real Images ---")
    
    # Create a base image (flat color to make signal obvious)
    base_img = np.full((128, 128, 3), 128, dtype=np.uint8)
    message = "hi" # Very short message
    
    import steganography.embed
    steganography.embed.ALPHA = 4
    
    print("\n[ALPHA=4] Testing short message 'hi'...")
    stego_4 = embed_message(base_img, message)
    cv2.imwrite("stego_4_hi.png", stego_4, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    label_4, conf_4 = predict_image("stego_4_hi.png")
    print(f"Result (ALPHA=4, msg='hi'): {label_4} ({conf_4:.2f}%)")
    
    # CASE: Increase ALPHA
    steganography.embed.ALPHA = 16
    print("\n[ALPHA=16] Testing short message 'hi'...")
    stego_16 = embed_message(base_img, message)
    cv2.imwrite("stego_16_hi.png", stego_16, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    label_16, conf_16 = predict_image("stego_16_hi.png")
    print(f"Result (ALPHA=16, msg='hi'): {label_16} ({conf_16:.2f}%)")
    
    # CASE: Long message with ALPHA=4
    steganography.embed.ALPHA = 4
    long_msg = "A" * 200
    print("\n[ALPHA=4] Testing long message (200 bytes)...")
    stego_long = embed_message(base_img, long_msg)
    cv2.imwrite("stego_long.png", stego_long, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    label_long, conf_long = predict_image("stego_long.png")
    print(f"Result (ALPHA=4, msg_len=200): {label_long} ({conf_long:.2f}%)")

if __name__ == "__main__":
    test_short_message_detection()
