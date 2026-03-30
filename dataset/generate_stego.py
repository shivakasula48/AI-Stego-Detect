import os
import sys
import glob
import random
import string
import logging

try:
    import cv2
    import numpy as np
except ImportError as e:
    sys.exit(f"Error: {e}\nPlease install dependencies using pip install -r requirements.txt inside a virtual environment")

logger = logging.getLogger(__name__)

from steganography.embed import embed_message
from encryption.aes import encrypt_message

# Fixed seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

from dataset.image_utils import create_gradient_image, create_pattern_image
import urllib.request


def generate_random_message(seed=None):
    if seed is not None:
        random.seed(seed)
    length = random.randint(10, 200)
    msg = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=length))
    random.seed()  # Reset
    return msg


def augment_image(img):
    """Apply random augmentations: flip, rotate, brightness, noise."""
    # Horizontal flip
    if random.random() > 0.5:
        img = cv2.flip(img, 1)

    # Rotation ±15 degrees
    angle = random.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Brightness / contrast variation
    alpha = random.uniform(0.8, 1.2)   # contrast
    beta = random.randint(-15, 15)     # brightness
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # Low-level Gaussian noise
    noise = np.random.normal(0, 3, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def download_sample_images(dest_dir, num=10):
    urls = [
        "https://images.pexels.com/photos/34950/pexels-photo.jpg",
        "https://images.pexels.com/photos/355465/pexels-photo-355465.jpeg",
        "https://images.pexels.com/photos/414612/pexels-photo-414612.jpeg",
        "https://images.pexels.com/photos/417173/pexels-photo-417173.jpeg",
        "https://images.pexels.com/photos/459225/pexels-photo-459225.jpeg",
        "https://images.pexels.com/photos/674010/pexels-photo-674010.jpeg",
        "https://images.pexels.com/photos/349758/pexels-photo-349758.jpeg",
        "https://images.pexels.com/photos/248797/pexels-photo-248797.jpeg",
    ]
    os.makedirs(dest_dir, exist_ok=True)
    for i, url in enumerate(urls[:num]):
        try:
            out_path = os.path.join(dest_dir, f"sample_{i+1}.jpg")
            urllib.request.urlretrieve(url, out_path)
            logger.info(f"Downloaded sample image {i+1}")
        except Exception as e:
            logger.error(f"Download failed for {url}: {e}")


def prepare_base_images(source_dir, num_images=100, size=(128, 128)):
    """Returns a list of (numpy image) ready for use as source material."""
    images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    if not images:
        logger.info("No real images found. Downloading samples...")
        download_sample_images(source_dir, num=8)
        images = [f for f in os.listdir(source_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    result = []
    if images:
        for fname in images:
            img = cv2.imread(os.path.join(source_dir, fname), cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = cv2.resize(img, size)
            result.append(img)
    else:
        logger.info("No real images available. Using synthetic generators.")

    # Fill remaining with synthetic if needed
    while len(result) < num_images:
        i = len(result)
        if i % 2 == 0:
            img = create_gradient_image(size, direction='horizontal' if i % 4 == 0 else 'vertical')
        else:
            img = create_pattern_image(size)
        result.append(img)

    return result[:num_images]


def generate_stego_dataset(
    base_dir="dataset",
    num_images=200,
    size=(128, 128),
    val_split=0.2,
    input_dir=None
):
    """
    Generates a balanced clean/stego dataset with augmentation,
    split into train/ and val/ sub-directories.

    Args:
        input_dir: Optional path to a folder of real images (.jpg/.jpeg/.png).
                   Falls back to synthetic generation if None or empty.
    """
    import shutil
    # Directory setup
    train_clean = os.path.join(base_dir, "train", "clean")
    train_stego = os.path.join(base_dir, "train", "stego")
    val_clean   = os.path.join(base_dir, "val",   "clean")
    val_stego   = os.path.join(base_dir, "val",   "stego")

    # Clear previous runs to ensure strict 1:1 balance
    for d in [os.path.join(base_dir, "train"), os.path.join(base_dir, "val")]:
        if os.path.exists(d):
            shutil.rmtree(d)
            logger.info(f"Cleared existing dataset directory: {d}")

    for d in [train_clean, train_stego, val_clean, val_stego]:
        os.makedirs(d, exist_ok=True)

    # Try loading real images first
    base_images = []
    if input_dir and os.path.isdir(input_dir):
        logger.info(f"Using real images from: {input_dir}")
        import glob
        paths = (
            glob.glob(os.path.join(input_dir, "*.jpg")) +
            glob.glob(os.path.join(input_dir, "*.jpeg")) +
            glob.glob(os.path.join(input_dir, "*.png")) +
            glob.glob(os.path.join(input_dir, "*.JPG")) +
            glob.glob(os.path.join(input_dir, "*.JPEG")) +
            glob.glob(os.path.join(input_dir, "*.PNG"))
        )
        # Remove duplicates (case-insensitive filesystems)
        paths = list(dict.fromkeys(paths))
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 4:
                img = img[:, :, :3]
            img = cv2.resize(img, size)
            base_images.append(img)
        logger.info(f"Loaded {len(base_images)} real images.")

    # Fallback: use synthetic + source_images pipeline
    if not base_images:
        if input_dir:
            logger.warning("No real images found. Using synthetic images.")
        source_dir = os.path.join(base_dir, "source_images")
        os.makedirs(source_dir, exist_ok=True)
        logger.info(f"Preparing {num_images} base images (synthetic)...")
        base_images = prepare_base_images(source_dir, num_images=num_images, size=size)

    # Oversample if we have fewer images than requested
    if len(base_images) < num_images:
        logger.info(f"Oversampling {len(base_images)} images to {num_images} with augmentation...")
        original = base_images.copy()
        while len(base_images) < num_images:
            img = random.choice(original).copy()
            base_images.append(augment_image(img))

    base_images = base_images[:num_images]
    logger.info(f"Got {len(base_images)} base images.")

    # Shuffle with fixed seed
    indices = list(range(len(base_images)))
    random.seed(SEED)
    random.shuffle(indices)

    # Debug directory for difference images
    debug_dir = os.path.join(base_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    n_val = max(1, int(len(indices) * val_split))
    n_train = len(indices) - n_val
    train_indices = indices[:n_train]
    val_indices   = indices[n_train:]

    def save_pair(img_list, clean_dir, stego_dir, prefix):
        for i, idx in enumerate(img_list):
            # --- AUGMENT FIRST ---
            # Any spatial augmentation (rotation, flipping) must happen BEFORE embedding
            # to keep the DCT grid pristine for the CNN.
            base_img = base_images[idx].copy()
            augmented_base = augment_image(base_img)

            # --- CLEAN ---
            clean_path = os.path.join(clean_dir, f"{prefix}_{i+1:04d}.png")
            cv2.imwrite(clean_path, augmented_base, [cv2.IMWRITE_PNG_COMPRESSION, 0])

            # --- STEGO ---
            msg = generate_random_message()
            stego_path = os.path.join(stego_dir, f"{prefix}_{i+1:04d}_stego.png")
            
            # Embed directly into the augmented image using unified function
            try:
                stego_img = embed_message(augmented_base, msg)
                cv2.imwrite(stego_path, stego_img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                ok = True
            except Exception as e:
                logger.error(f"Failed to embed stego for {prefix}_{i+1}: {e}")
                ok = False

            if (i + 1) % 10 == 0:
                logger.info(f"[{prefix}] Progress: {i+1}/{len(img_list)}")

    logger.info(f"Generating {len(train_indices)} training pairs...")
    save_pair(train_indices, train_clean, train_stego, "train")

    logger.info(f"Generating {len(val_indices)} validation pairs...")
    save_pair(val_indices, val_clean, val_stego, "val")

    # Stats
    tc = len(os.listdir(train_clean))
    ts = len(os.listdir(train_stego))
    vc = len(os.listdir(val_clean))
    vs = len(os.listdir(val_stego))
    logger.info(f"\nDataset ready:")
    logger.info(f"  Train → Clean: {tc}, Stego: {ts}")
    logger.info(f"  Val   → Clean: {vc}, Stego: {vs}")

    # Keep flat clean/stego dirs also updated (for backward compat)
    flat_clean = os.path.join(base_dir, "clean")
    flat_stego = os.path.join(base_dir, "stego")
    return train_clean, train_stego, val_clean, val_stego
