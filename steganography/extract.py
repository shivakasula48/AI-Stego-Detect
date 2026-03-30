import os
import sys
import logging

try:
    import cv2
    import numpy as np
except ImportError as e:
    sys.exit(f"Error: {e}\nPlease install dependencies using pip install -r requirements.txt inside a virtual environment")

logger = logging.getLogger(__name__)


def extract_data(image_path) -> bytes:
    """
    Extracts embedded data bytes from DCT coefficients across BGR channels.
    Uses a 4-byte length header to determine message size.
    """
    if not os.path.exists(image_path):
        logger.error(f" Image not found: {image_path}")
        return b""

    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        logger.error(" Invalid image or unsupported format.")
        return b""

    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    h, w, c = image.shape
    channels = cv2.split(image)  # [B, G, R]
    channels_f32 = [np.float32(ch) for ch in channels]

    # All 32 mid-frequency AC coefficients (match embed.py order exactly)
    ZIGZAG_FULL = [
        (0,1), (1,0), (2,0), (1,1), (0,2), (0,3), (1,2), (2,1),
        (3,0), (4,0), (3,1), (2,2), (1,3), (0,4), (0,5), (1,4),
        (2,3), (3,2), (4,1), (5,0), (6,0), (5,1), (4,2), (3,3),
        (2,4), (1,5), (0,6), (0,7), (1,6), (2,5), (3,4), (4,3)
    ]
    AC_COEFFS = ZIGZAG_FULL
    Q = 48

    length_bytes = bytearray()
    extracted_bytes = bytearray()
    current_byte = 0
    bit_count = 0
    total_bits = 0
    data_length = None
    expected_bits = -1

    # Process channels (B, G, R) in the exact same order as embed.py
    for ch_idx in range(c):
        ch_data = channels_f32[ch_idx]
        
        for row in range(0, h - (h % 8), 8):
            for col in range(0, w - (w % 8), 8):
                block = ch_data[row:row+8, col:col+8]
                dct_block = cv2.dct(block)
                
                for u, v in AC_COEFFS:
                    coef = dct_block[u, v]
                    # Quantization recovery
                    step = round(coef / Q)
                    bit = int(step % 2)
                    
                    # Shift bit into byte (MSB first)
                    current_byte = (current_byte << 1) | bit
                    bit_count += 1
                    total_bits += 1
                    
                    if bit_count == 8:
                        if total_bits <= 32:
                            length_bytes.append(current_byte)
                            if total_bits == 32:
                                data_length = int.from_bytes(bytes(length_bytes), byteorder='big')
                                expected_bits = 32 + (data_length * 8)
                        else:
                            extracted_bytes.append(current_byte)
                        
                        current_byte = 0
                        bit_count = 0

                    if expected_bits != -1 and total_bits >= expected_bits:
                        break
                if expected_bits != -1 and total_bits >= expected_bits:
                    break
            if expected_bits != -1 and total_bits >= expected_bits:
                break
        if expected_bits != -1 and total_bits >= expected_bits:
            break

    if data_length is None or total_bits < expected_bits:
        raise Exception(f"Extraction failed: Expected {expected_bits} bits, got {total_bits}.")
    
    return bytes(extracted_bytes)


if __name__ == "__main__":
    from steganography.embed import run_self_test
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    run_self_test()
