import cv2
import numpy as np
import sys
sys.path.insert(0, '/home/shiva/Desktop/takur')

from steganography.embed import embed_message
from steganography.extract import extract_data
from encryption.aes import decrypt_message

# Create a plain white image (255, 255, 255) to simulate a white flow diagram
img = np.ones((128, 128, 3), dtype=np.uint8) * 255
msg = "FINALLY DETECTABLE"

try:
    stego1 = embed_message(img, msg)
    cv2.imwrite('/tmp/stego1.png', stego1)
    extracted1 = extract_data('/tmp/stego1.png')
    dec1 = decrypt_message(extracted1) if extracted1 else "Failed extraction"
    print(f"White Image Default -> {dec1}")
except Exception as e:
    print(f"White Image Default -> Error: {e}")

img_clamped = np.clip(img.astype(np.int32), 15, 240).astype(np.uint8)
try:
    stego2 = embed_message(img_clamped, msg)
    cv2.imwrite('/tmp/stego2.png', stego2)
    extracted2 = extract_data('/tmp/stego2.png')
    dec2 = decrypt_message(extracted2) if extracted2 else "Failed extraction"
    print(f"White Image Clamped -> {dec2}")
except Exception as e:
    print(f"White Image Clamped -> Error: {e}")
