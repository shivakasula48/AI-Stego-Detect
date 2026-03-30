#!/usr/bin/env python3
"""Debug roundtrip: embed -> save -> extract -> decrypt"""
import os
import sys
sys.path.insert(0, '/home/shiva/Desktop/takur')

import cv2
import numpy as np

from steganography.embed import embed_message
from steganography.extract import extract_data
from encryption.aes import decrypt_message, encrypt_message

def test_roundtrip(image_path, message, out_path='/tmp/debug_stego_out.png'):
    print(f"\n{'='*50}")
    print(f"Input image: {image_path}")
    print(f"Message:     '{message}'")

    # Load
    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Cannot read {image_path}")
        return

    print(f"Original size: {img.shape}")

    # Resize to 128x128 (as app.py does)
    img_128 = cv2.resize(img, (128, 128))
    print(f"After resize: {img_128.shape}")

    # Embed
    stego = embed_message(img_128, message)
    print(f"Stego shape:  {stego.shape}")

    # Save losslessly
    cv2.imwrite(out_path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Saved to:     {out_path}")

    # --- Simulate extract route ---
    img_reload = cv2.imread(out_path)
    print(f"Reloaded shape: {img_reload.shape}")
    print(f"Pixel diff (stego vs reload): {np.abs(stego.astype(int) - img_reload.astype(int)).max()}")

    # Extract raw bytes
    enc_bytes = extract_data(out_path)
    print(f"Extracted bytes length: {len(enc_bytes)}")

    if not enc_bytes:
        print("FAIL: No bytes extracted")
        return

    # Verify vs original encrypted bytes
    enc_original = encrypt_message(message)
    print(f"Original enc length: {len(enc_original)}")

    # Decrypt
    result = decrypt_message(enc_bytes)
    print(f"Decrypted result: '{result}'")

    if result == message:
        print("✅ ROUNDTRIP SUCCESS")
    else:
        print("❌ ROUNDTRIP FAILED")

if __name__ == '__main__':
    # Test with a natural photo
    test_roundtrip('images/img_0002.jpg', 'nani')
    # Test with another image
    test_roundtrip('images/img_0005.jpg', 'Hello World Test 12345')
