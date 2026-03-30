import os
import sys
import logging

try:
    import cv2
    import numpy as np
except ImportError as e:
    sys.exit(f"Error: {e}\nPlease install dependencies using pip install -r requirements.txt inside a virtual environment")

logger = logging.getLogger(__name__)


def _string_to_bin(data):
    """Convert bytes to a binary string."""
    return ''.join(f'{byte:08b}' for byte in data)


def generate_diff_image(original, stego, output_path):
    """Saves a magnified difference image for debugging."""
    diff = cv2.absdiff(original, stego)
    # Magnify for visibility (e.g. 10x)
    magnified = np.clip(diff.astype(np.float32) * 10, 0, 255).astype(np.uint8)
    cv2.imwrite(output_path, magnified)
    logger.debug(f" Difference image saved to: {output_path}")


def embed_data(image_path, data_bytes, output_path, diff_path=None):
    """
    Embeds data_bytes into the image at image_path using DCT on all BGR channels.
    Saves the result at output_path.
    If diff_path is provided, saves a magnified difference image.
    """
    if not os.path.exists(image_path):
        logger.error(f" Image not found: {image_path}")
        return False

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        logger.error(" Invalid image or unsupported format.")
        return False

    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    h, w, c = image.shape
    channels = cv2.split(image)  # [B, G, R]
    channels_f32 = [np.float32(ch) for ch in channels]

    data_length = len(data_bytes)
    length_bytes = data_length.to_bytes(4, byteorder='big')
    final_data = length_bytes + data_bytes
    binary_data = _string_to_bin(final_data)
    data_len = len(binary_data)

    # Standard ZigZag order (excluding DC at 0,0)
    # We pick the first 32 stable mid-frequency coefficients
    ZIGZAG_FULL = [
        (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2), (2,1),
        (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5), (1,4),
        (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2), (3,3),
        (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4), (4,3)
    ]
    AC_COEFFS = ZIGZAG_FULL
    
    # Capacity Check
    max_capacity = (h // 8) * (w // 8) * c * len(AC_COEFFS)
    if data_len > max_capacity:
        logger.error(f" Data too large ({data_len} bits) for image capacity ({max_capacity} bits).")
        return False

    bit_idx = 0
    Q = 48      # Increased for extreme robustness
    ALPHA = 4   # Noticeable perturbation for CNN patterns

    # Process each channel (B, G, R)
    for ch_idx in range(c):
        ch_data = channels_f32[ch_idx]
        
        for row in range(0, h - (h % 8), 8):
            for col in range(0, w - (w % 8), 8):
                if bit_idx >= data_len:
                    break
                    
                block = ch_data[row:row+8, col:col+8]
                dct_block = cv2.dct(block)
                
                for u, v in AC_COEFFS:
                    if bit_idx >= data_len:
                        break
                        
                    bit = int(binary_data[bit_idx])
                    coef = dct_block[u, v]
                    
                    # QIM embedding
                    step = round(coef / Q)
                    if (step % 2) != bit:
                        if coef > step * Q:
                            step += 1
                        else:
                            step -= 1
                            
                    # Apply final step * Q + ALPHA shift for amplification
                    shift = ALPHA if bit == 1 else -ALPHA
                    dct_block[u, v] = (step * Q) + shift
                    
                    bit_idx += 1
                    
                ch_data[row:row+8, col:col+8] = cv2.idct(dct_block)
                
            if bit_idx >= data_len:
                break
        if bit_idx >= data_len:
            break

    # Reconstruct Image
    merged_channels = [np.clip(np.round(ch), 0, 255).astype(np.uint8) for ch in channels_f32]
    stego_bgr = cv2.merge(merged_channels)
    
    cv2.imwrite(output_path, stego_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    
    # Generate difference image for debugging purposes if provided
    if diff_path:
        generate_diff_image(image, stego_bgr, diff_path)
    
    return True


def run_self_test():
    """
    Runs a full pipeline test: encrypt -> embed -> extract -> decrypt.
    """
    from steganography.extract import extract_data
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../encryption')))
    try:
        from encryption.aes import encrypt_message, decrypt_message
    except ImportError:
        logger.error("[FAIL] Could not import encryption module.")
        return

    dummy_img_path = "pipeline_test_img.png"
    if not os.path.exists(dummy_img_path):
        dummy = np.random.randint(0, 256, (128, 128, 3), dtype=np.uint8)
        cv2.imwrite(dummy_img_path, dummy)

    message = "Testing enhanced stego signal with RGB + ALPHA"
    print(f"Testing message: {message}")
    
    enc = encrypt_message(message)
    stego_path = "pipeline_test_stego.png"
    
    if embed_data(dummy_img_path, enc, stego_path):
        extracted = extract_data(stego_path)
        dec = decrypt_message(extracted)
        if dec == message:
             print("✅ ROUNDTRIP SUCCESSFUL: Enhanced signal extraction verified.")
        else:
             print(f"❌ ROUNDTRIP FAILED: Decrypted as '{dec}'")
             sys.exit(1)
    else:
        print("❌ EMBEDDING FAILED")
        sys.exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    run_self_test()
