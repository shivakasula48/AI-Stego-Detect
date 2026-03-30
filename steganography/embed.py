import os
import sys
import logging

try:
    import cv2
    import numpy as np
except ImportError as e:
    sys.exit(f"Error: {e}\nPlease install dependencies using pip install -r requirements.txt inside a virtual environment")

logger = logging.getLogger(__name__)


from encryption.aes import encrypt_message

# Core DCT Embedding Parameters
Q = 48        # Step size for QIM
ALPHA = 4     # Robustness boost (noticeable for CNN)
ZIGZAG_32 = [
    (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2), (2,1),
    (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5), (1,4),
    (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2), (3,3),
    (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4), (4,3)
]

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


def embed_message(image: np.ndarray, message: str) -> np.ndarray:
    """
    Encrypts a message and embeds it into the given BGR image using DCT.
    Returns: The stego image as a numpy array.
    """
    # Ensure image is in common format (BGR uint8)
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
        
    # Prevent DCT overflow/underflow on uniform areas (like white backgrounds)
    # by clamping pixel values to [15, 240] before embedding to allow +/- shift.
    image = np.clip(image.astype(np.int32), 15, 240).astype(np.uint8)
    
    stego = image.copy().astype(np.float32)

    # 1. Encrypt Message
    data_bytes = encrypt_message(message)
    
    # 2. Add Length Header (4 bytes)
    full_payload = len(data_bytes).to_bytes(4, byteorder='big') + data_bytes
    binary_data = _string_to_bin(full_payload)
    data_len = len(binary_data)

    h, w, c = stego.shape
    max_capacity = (h // 8) * (w // 8) * c * len(ZIGZAG_32)
    
    if data_len > max_capacity:
        raise ValueError(f"Message too large ({data_len} bits) for image capacity ({max_capacity} bits).")

    bit_idx = 0
    channels = cv2.split(stego)

    # 3. Embed across BGR channels
    for ch_idx in range(c):
        ch_data = channels[ch_idx]
        for row in range(0, h - (h % 8), 8):
            for col in range(0, w - (w % 8), 8):
                if bit_idx >= data_len: break
                    
                block = ch_data[row:row+8, col:col+8]
                dct_block = cv2.dct(block)
                
                for u, v in ZIGZAG_32:
                    if bit_idx >= data_len: break
                        
                    bit = int(binary_data[bit_idx])
                    coef = dct_block[u, v]
                    
                    # QIM embedding
                    step = round(coef / Q)
                    if (step % 2) != bit:
                        step += 1 if coef > step * Q else -1
                            
                    # Apply ALPHA shift
                    shift = ALPHA if bit == 1 else -ALPHA
                    dct_block[u, v] = (step * Q) + shift
                    bit_idx += 1
                    
                ch_data[row:row+8, col:col+8] = cv2.idct(dct_block)
            if bit_idx >= data_len: break
        if bit_idx >= data_len: break

    # Final Merge
    merged = [np.clip(np.round(ch), 0, 255).astype(np.uint8) for ch in channels]
    return cv2.merge(merged)


def embed_data(image_path, data_bytes, output_path, diff_path=None):
    """
    Legacy wrapper for file-based embedding. 
    Note: Now bypasses internal encryption because it receives pre-encrypted bytes.
    Use embed_message for the unified pipeline.
    """
    image = cv2.imread(image_path)
    if image is None: return False

    # Prevent DCT overflow/underflow
    image = np.clip(image.astype(np.int32), 15, 240).astype(np.uint8)

    # Since this function is for legacy support and takes bytes, we replicate the 
    # logic or wrap a raw version. For now, let's keep it robust but using the same constants.
    full_payload = len(data_bytes).to_bytes(4, byteorder='big') + data_bytes
    binary_data = _string_to_bin(full_payload)
    data_len = len(binary_data)

    h, w, c = image.shape
    stego = image.astype(np.float32)
    channels = cv2.split(stego)
    bit_idx = 0

    for ch_idx in range(c):
        ch_data = channels[ch_idx]
        for row in range(0, h - (h % 8), 8):
            for col in range(0, w - (w % 8), 8):
                if bit_idx >= data_len: break
                block = ch_data[row:row+8, col:col+8]
                dct_block = cv2.dct(block)
                for u, v in ZIGZAG_32:
                    if bit_idx >= data_len: break
                    bit = int(binary_data[bit_idx])
                    step = round(dct_block[u, v] / Q)
                    if (step % 2) != bit:
                        step += 1 if dct_block[u, v] > step * Q else -1
                    dct_block[u, v] = (step * Q) + (ALPHA if bit == 1 else -ALPHA)
                    bit_idx += 1
                ch_data[row:row+8, col:col+8] = cv2.idct(dct_block)
            if bit_idx >= data_len: break
        if bit_idx >= data_len: break

    res = cv2.merge([np.clip(np.round(ch), 0, 255).astype(np.uint8) for ch in channels])
    cv2.imwrite(output_path, res, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    if diff_path:
        generate_diff_image(image, res, diff_path)
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
    
    if embed_data(dummy_img_path, encrypt_message(message), stego_path):
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
