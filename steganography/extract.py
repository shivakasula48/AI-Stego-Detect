import os
import sys
import logging

try:
    import cv2
    import numpy as np
except ImportError as e:
    sys.exit(f"Error: {e}\nPlease install dependencies using pip install -r requirements.txt inside a virtual environment")

from steganography.embed import Q, ZIGZAG_32
from encryption.aes import decrypt_message

logger = logging.getLogger(__name__)


def extract_message(image: np.ndarray) -> str:
    """
    Extracts and decrypts a message from a stego image array.
    Returns: The decrypted string or error message.
    """
    if len(image.shape) == 3 and image.shape[2] == 4:
        image = image[:, :, :3]
    
    h, w, c = image.shape
    channels = cv2.split(image.astype(np.float32))
    
    length_bytes = bytearray()
    extracted_bytes = bytearray()
    current_byte = 0
    bit_count = 0
    total_bits = 0
    data_length = None
    expected_bits = -1

    for ch_idx in range(c):
        ch_data = channels[ch_idx]
        for row in range(0, h - (h % 8), 8):
            for col in range(0, w - (w % 8), 8):
                block = ch_data[row:row+8, col:col+8]
                dct_block = cv2.dct(block)
                for u, v in ZIGZAG_32:
                    coef = dct_block[u, v]
                    bit = int(round(coef / Q) % 2)
                    
                    current_byte = (current_byte << 1) | bit
                    bit_count += 1
                    total_bits += 1
                    
                    if bit_count == 8:
                        if total_bits <= 32:
                            length_bytes.append(current_byte)
                            if total_bits == 32:
                                data_length = int.from_bytes(bytes(length_bytes), byteorder='big')
                                # Security/Sanity check for length
                                if data_length > 1000000: # 1MB limit
                                    return "[ERROR] Invalid length header detected."
                                expected_bits = 32 + (data_length * 8)
                        else:
                            extracted_bytes.append(current_byte)
                        current_byte = 0
                        bit_count = 0

                    if expected_bits != -1 and total_bits >= expected_bits:
                        break
                if expected_bits != -1 and total_bits >= expected_bits: break
            if expected_bits != -1 and total_bits >= expected_bits: break
        if expected_bits != -1 and total_bits >= expected_bits: break

    if data_length is None or total_bits < expected_bits:
        return "[ERROR] Extraction failed: Data incomplete."
    
    return decrypt_message(bytes(extracted_bytes))


def extract_data(image_path) -> bytes:
    """
    Legacy wrapper for file-based extraction. 
    Returns raw encrypted bytes.
    """
    image = cv2.imread(image_path)
    if image is None: return b""
    
    h, w, c = image.shape
    channels = cv2.split(image.astype(np.float32))
    
    length_bytes = bytearray()
    extracted_bytes = bytearray()
    current_byte = 0
    bit_count = 0
    total_bits = 0
    data_length = None
    expected_bits = -1

    for ch_idx in range(c):
        ch_data = channels[ch_idx]
        for row in range(0, h - (h % 8), 8):
            for col in range(0, w - (w % 8), 8):
                dct_block = cv2.dct(ch_data[row:row+8, col:col+8])
                for u, v in ZIGZAG_32:
                    bit = int(round(dct_block[u, v] / Q) % 2)
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
                    if expected_bits != -1 and total_bits >= expected_bits: break
                if expected_bits != -1 and total_bits >= expected_bits: break
            if expected_bits != -1 and total_bits >= expected_bits: break
        if expected_bits != -1 and total_bits >= expected_bits: break

    return bytes(extracted_bytes)


if __name__ == "__main__":
    from steganography.embed import run_self_test
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    run_self_test()
