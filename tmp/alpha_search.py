import cv2
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from steganography.embed import embed_message
from ai_model.predict import predict_image

def final_alpha_search():
    print("--- Exhaustive ALPHA Search for Textured Image Detection ---")
    
    # Create a complex textured image (simulating a real-world edge-heavy image)
    np.random.seed(42)
    img = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
    message = "Testing signal strength for robust detection."
    
    import steganography.embed
    results = []
    
    for alpha in [4, 8, 16, 32, 48, 64]:
        steganography.embed.ALPHA = alpha
        stego = embed_message(img, message)
        path = f"stego_alpha_{alpha}.png"
        cv2.imwrite(path, stego, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        label, conf = predict_image(path)
        results.append((alpha, label, conf))
        print(f"ALPHA={alpha:2d} | Result: {label:12s} | Confidence: {conf:6.2f}%")
        os.remove(path)

    print("\n--- Summary ---")
    for a, l, c in results:
        print(f"Alpha {a}: {l} ({c:.2f}%)")

if __name__ == "__main__":
    final_alpha_search()
