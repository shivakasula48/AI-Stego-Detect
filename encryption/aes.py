import sys
import logging

try:
    from Crypto.Cipher import AES
    from Crypto.Protocol.KDF import PBKDF2
    from Crypto.Random import get_random_bytes
    from Crypto.Util.Padding import pad, unpad
except ImportError as e:
    sys.exit(f"Error: {e}\nPlease install dependencies using pip install -r requirements.txt inside a virtual environment")

logger = logging.getLogger(__name__)
import base64

# Hardcoded password for ease of use in the CLI. 
# Alternatively, could be passed as an argument.
PASSWORD = b"stego_master_key_2026"
SALT = b"stationary_salt_for_demo_project_only"

def _get_key():
    """Derive a 32-byte AES key from a given password and salt."""
    key = PBKDF2(PASSWORD, SALT, dkLen=32, count=100000)
    return key

def encrypt_message(message: str) -> bytes:
    """
    Encrypts a string message using AES-CBC.
    Returns the iv and ciphertext concatenated.
    """
    key = _get_key()
    
    # Generate random IV
    iv = get_random_bytes(16)
    
    # Create cipher
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    # Encrypt
    ciphertext = cipher.encrypt(pad(message.encode('utf-8'), AES.block_size))
    
    # Return
    return iv + ciphertext   # IMPORTANT

def decrypt_message(cipher_bytes: bytes) -> str:
    """
    Decrypts the cipher_bytes array using AES-CBC.
    Extracts IV, and returns the original message.
    """
    key = _get_key()
    
    if len(cipher_bytes) < 16:
        return "[ERROR] Payload too short to contain IV"

    # Extract IV
    iv = cipher_bytes[:16]
    ciphertext = cipher_bytes[16:]
    
    # Create cipher
    cipher = AES.new(key, AES.MODE_CBC, iv)
    
    try:
        # Decrypt
        message = unpad(cipher.decrypt(ciphertext), AES.block_size)
        return message.decode('utf-8')
    except ValueError as e:
        return f"[ERROR] Decryption or unpadding failed: {e}"


# ================= Self-Test System ===================
def run_self_test():
    """
    Runs a suite of encryption/decryption tests and prints results.
    Raises Exception if any test fails.
    """
    test_messages = [
        "Hello",
        "CyberSecurity123",
        "Steganography Test",
        "1234567890",
        "",  # Empty string
        "A" * 10000  # Long message
    ]
    all_passed = True
    for idx, msg in enumerate(test_messages):
        try:
            enc = encrypt_message(msg)
            dec = decrypt_message(enc)
            if dec == msg:
                logger.info(f"[PASS] Test {idx+1}: '{msg[:30] + ('...' if len(msg) > 30 else '')}'")
            else:
                logger.error(f"[FAIL] Test {idx+1}: '{msg[:30] + ('...' if len(msg) > 30 else '')}'")
                print(f"        Decrypted: '{dec}'")
                all_passed = False
        except Exception as e:
            logger.error(f"[FAIL] Test {idx+1}: Exception occurred: {e}")
            all_passed = False
    if all_passed:
        print("\nAll encryption/decryption tests PASSED.")
    else:
        raise Exception("One or more encryption/decryption tests FAILED. See above for details.")


# Run self-test if executed directly
if __name__ == "__main__":
    run_self_test()

    logger.info("\n--- STRICT AES VERIFICATION ---")
    message = "TEST_MESSAGE_123456"
    logger.info(f"Original: {message}")
    
    encrypted_bytes = encrypt_message(message)
    logger.info(f"Encrypted bytes length: {len(encrypted_bytes)}")
    
    decrypted = decrypt_message(encrypted_bytes)
    logger.info(f"Decrypted: {decrypted}")
    
    if message != decrypted:
        raise Exception("AES BROKEN")
    else:
        logger.info("AES Strict Verification Passed.")
