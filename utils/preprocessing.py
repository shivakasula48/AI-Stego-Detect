import os
import cv2
import numpy as np

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    Reads an image from disk, converts to RGB (cv2 default is BGR), 
    resizes to target_size (if needed), and normalizes pixel values to [0, 1].
    """
    image = cv2.imread(image_path)
    if image is None:
        return None
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Matching training: Skip resize if already correct size to avoid noise
    h, w = image.shape[:2]
    if h != target_size[1] or w != target_size[0]:
        image = cv2.resize(image, target_size)
        
    image = image.astype(np.float32) / 255.0
    return image

def load_dataset(clean_dir="dataset/clean", stego_dir="dataset/stego", target_size=(128, 128)):
    """
    Loads all clean and stego images.
    Returns X (images array) and y (labels array: 0 for clean, 1 for stego).
    """
    X = []
    y = []

    # Process Clean Images (Label 0)
    if os.path.exists(clean_dir):
        for filename in os.listdir(clean_dir):
            img_path = os.path.join(clean_dir, filename)
            img = load_and_preprocess_image(img_path, target_size)
            if img is not None:
                X.append(img)
                y.append(0)
    else:
        print(f"[Warning] Clean directory '{clean_dir}' not found.")

    # Process Stego Images (Label 1)
    if os.path.exists(stego_dir):
        for filename in os.listdir(stego_dir):
            img_path = os.path.join(stego_dir, filename)
            img = load_and_preprocess_image(img_path, target_size)
            if img is not None:
                X.append(img)
                y.append(1)
    else:
        print(f"[Warning] Stego directory '{stego_dir}' not found.")

    # Convert to numpy arrays if data exists
    if not X:
        return np.array([]), np.array([])
        
    return np.array(X), np.array(y)
