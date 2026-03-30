import argparse
import logging
import sys
import os

from steganography.embed import embed_data
from steganography.extract import extract_data
from encryption.aes import encrypt_message, decrypt_message
from dataset.generate_stego import generate_stego_dataset

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(levelname)s] %(message)s'
    )

def handle_embed(args):
    logging.info("Encrypting message...")
    enc_bytes = encrypt_message(args.message)
    if not enc_bytes:
        logging.error("Failed to encrypt message.")
        sys.exit(1)

    logging.info(f"Embedding into {args.input}...")
    
    # CRITICAL: CNN expects 128x128. Resize BEFORE embedding to preserve signal.
    import cv2
    img = cv2.imread(args.input)
    if img is None:
        logging.error(f"Could not read cover image: {args.input}")
        sys.exit(1)
        
    if img.shape[0] != 128 or img.shape[1] != 128:
        logging.info("Resizing cover image to 128x128 for AI compatibility...")
        img = cv2.resize(img, (128, 128))
    
    # Save temp resized image or use in-memory embedding
    # The current embed_data function reads from path, so we save a temporary one
    # or better: we update the logic to handle in-memory.
    # But for now, let's keep it simple and overwrite the path for embed_data 
    # OR create a temporary file.
    temp_cover = "tmp_cover_128.png"
    cv2.imwrite(temp_cover, img)
    
    success = embed_data(temp_cover, enc_bytes, args.output)
    if os.path.exists(temp_cover):
        os.remove(temp_cover)
        
    if success:
        logging.info(f"Successfully created stego image: {args.output} (Size: 128x128)")
    else:
        logging.error("Failed to embed data.")
        sys.exit(1)

def handle_extract(args):
    logging.info(f"Extracting from {args.input}...")
    enc_bytes = extract_data(args.input)
    if not enc_bytes:
        logging.error("Extraction returned empty data.")
        sys.exit(1)

    logging.info("Decrypting message...")
    msg = decrypt_message(enc_bytes)
    if msg and not msg.startswith("[ERROR]"):
        logging.info(f"Extracted Message: {msg}")
        return msg
    else:
        logging.error(f"Decryption failed: {msg}")
        sys.exit(1)

def handle_generate(args):
    input_src = f" from {args.input_dir}" if args.input_dir else " (synthetic)"
    logging.info(f"Generating stego dataset (base: {args.output}, images: {args.num_images}){input_src}")
    generate_stego_dataset(base_dir=args.output, num_images=args.num_images, input_dir=args.input_dir)
    logging.info("Dataset generation complete.")

def handle_detect(args):
    try:
        from ai_model.predict import predict_image
    except ImportError as e:
        logging.error(f"Could not import detection module: {e}")
        sys.exit(1)

    logging.info(f"Running AI detection on: {args.input}")
    label, confidence = predict_image(args.input)

    if label is None:
        logging.error("Detection failed. Ensure the model is trained and the image path is correct.")
        sys.exit(1)

    if label == "Stego Image":
        logging.info(f"[RESULT] Stego Image (Hidden Data Detected) | Confidence: {confidence:.2f}%")
    else:
        logging.info(f"[RESULT] Clean Image (No Hidden Data) | Confidence: {confidence:.2f}%")

def handle_train(args):
    try:
        from ai_model.train import train_model
    except ImportError as e:
        logging.error(f"Could not import training module: {e}")
        sys.exit(1)

    logging.info("Starting CNN model training...")
    train_model(model_save_path=args.output, base_dir=args.dataset)

def handle_optimize(args):
    """
    Full automation: generate dataset → train → evaluate → retry if needed.
    Repeats up to max_attempts times or until val_accuracy >= target.
    """
    import shutil
    import glob

    target_acc = args.target
    max_attempts = args.max_attempts
    num_images = args.num_images
    base_dir = args.dataset
    input_dir = args.input_dir
    model_path = "ai_model/stego_detector.h5"

    best_acc = 0.0

    for attempt in range(1, max_attempts + 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"OPTIMIZE — Attempt {attempt}/{max_attempts}")
        logging.info(f"{'='*60}")

        # 1. Clear previous dataset for fresh generation
        for sub in ["train", "val"]:
            p = os.path.join(base_dir, sub)
            if os.path.exists(p):
                shutil.rmtree(p)
                logging.info(f"Cleared {p}")

        # 2. Generate dataset with varying seed for diversity per attempt
        logging.info(f"Generating {num_images} image pairs (attempt seed offset={attempt})...")
        import dataset.generate_stego as dsg
        original_seed = dsg.SEED
        dsg.SEED = 42 + attempt  # Shift seed per attempt for new augmentations
        dsg.random.seed(dsg.SEED)
        dsg.np.random.seed(dsg.SEED)
        generate_stego_dataset(base_dir=base_dir, num_images=num_images, input_dir=input_dir)
        dsg.SEED = original_seed  # Restore

        # 3. Train
        try:
            from ai_model.train import train_model
        except ImportError as e:
            logging.error(f"Could not import training module: {e}")
            sys.exit(1)

        logging.info("Training model...")
        val_acc = train_model(model_save_path=model_path, base_dir=base_dir)

        if val_acc is None:
            logging.error("Training failed.")
            continue

        best_acc = max(best_acc, val_acc)
        logging.info(f"Attempt {attempt} — Validation Accuracy: {val_acc:.2f}%")

        if val_acc >= target_acc:
            logging.info(f"TARGET REACHED: {val_acc:.2f}% ≥ {target_acc}%")
            break
        else:
            logging.warning(f"Below target ({val_acc:.2f}% < {target_acc}%). Retrying...")

    # 4. Detection validation on held-out images
    logging.info(f"\n{'='*60}")
    logging.info("DETECTION VALIDATION")
    logging.info(f"{'='*60}")

    try:
        from ai_model.predict import predict_image
        from tensorflow.keras.models import load_model
        from utils.preprocessing import load_and_preprocess_image
        import numpy as np
    except ImportError:
        logging.error("Cannot import prediction modules for validation.")
        return

    # Load model once to avoid repeated loading / TF retracing
    model_file = "ai_model/stego_detector.h5"
    if not os.path.exists(model_file):
        logging.warning("Model file not found, skipping detection validation.")
        return

    model = load_model(model_file)

    val_clean_dir = os.path.join(base_dir, "val", "clean")
    val_stego_dir = os.path.join(base_dir, "val", "stego")

    def test_dir(directory, expected_label, max_samples=5):
        if not os.path.exists(directory):
            logging.warning(f"Directory not found: {directory}")
            return 0, 0
        files = [f for f in os.listdir(directory) if f.endswith('.png')][:max_samples]
        correct = 0
        for fname in files:
            fpath = os.path.join(directory, fname)
            img = load_and_preprocess_image(fpath, target_size=(128, 128))
            if img is None:
                continue
            pred = float(model.predict(np.expand_dims(img, 0), verbose=0)[0][0])
            label = "Stego Image" if pred >= 0.5 else "Clean Image"
            conf = pred * 100 if pred >= 0.5 else (1.0 - pred) * 100
            logging.info(f"  {fname}: {label} ({conf:.1f}%)")
            if label == expected_label:
                correct += 1
        return correct, len(files)

    clean_correct, clean_total = test_dir(val_clean_dir, "Clean Image", 5)
    stego_correct, stego_total = test_dir(val_stego_dir, "Stego Image", 5)

    clean_acc = (clean_correct / clean_total * 100) if clean_total > 0 else 0
    stego_acc = (stego_correct / stego_total * 100) if stego_total > 0 else 0

    # 5. Final report
    logging.info(f"\n{'='*60}")
    logging.info("OPTIMIZATION REPORT")
    logging.info(f"{'='*60}")
    logging.info(f"Best Validation Accuracy:  {best_acc:.2f}%")
    logging.info(f"Clean Detection Accuracy:  {clean_acc:.0f}% ({clean_correct}/{clean_total})")
    logging.info(f"Stego Detection Accuracy:  {stego_acc:.0f}% ({stego_correct}/{stego_total})")
    logging.info(f"Attempts Used:             {min(attempt, max_attempts)}/{max_attempts}")
    if best_acc >= target_acc:
        logging.info(f"STATUS: ✅ TARGET MET ({target_acc}%)")
    else:
        logging.warning(f"STATUS: ⚠️ TARGET NOT MET (best={best_acc:.2f}%, target={target_acc}%)")
    logging.info(f"Model saved to:            {model_path}")
    logging.info(f"{'='*60}")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="AI-Based Encrypted Image Steganography Detection System")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate-dataset
    parser_gen = subparsers.add_parser("generate-dataset", help="Generate clean and stego images for AI training")
    parser_gen.add_argument("--output", type=str, default="dataset", help="Base dataset directory (will contain train/ and val/)")
    parser_gen.add_argument("--num-images", type=int, default=200, help="Number of image pairs to generate")
    parser_gen.add_argument("--input-dir", type=str, default=None, help="Path to folder of real images (.jpg/.png)")

    # embed
    parser_embed = subparsers.add_parser("embed", help="Embed an encrypted message into an image")
    parser_embed.add_argument("--input", required=True, type=str, help="Input cover image path")
    parser_embed.add_argument("--message", required=True, type=str, help="Secret message to embed")
    parser_embed.add_argument("--output", required=True, type=str, help="Output stego image path")

    # extract
    parser_extract = subparsers.add_parser("extract", help="Extract and decrypt a message from a stego image")
    parser_extract.add_argument("--input", required=True, type=str, help="Input stego image path")

    # detect
    parser_detect = subparsers.add_parser("detect", help="Detect whether an image contains hidden data using AI")
    parser_detect.add_argument("--input", required=True, type=str, help="Input image path to analyse")

    # train
    parser_train = subparsers.add_parser("train", help="Train the CNN steganalysis model")
    parser_train.add_argument("--output", type=str, default="ai_model/stego_detector.h5", help="Path to save trained model")
    parser_train.add_argument("--dataset", type=str, default="dataset", help="Base dataset directory containing train/ and val/")

    # optimize
    parser_opt = subparsers.add_parser("optimize", help="Auto-generate dataset, train, evaluate, and retry until target accuracy")
    parser_opt.add_argument("--num-images", type=int, default=400, help="Number of image pairs per attempt")
    parser_opt.add_argument("--target", type=float, default=75.0, help="Target validation accuracy")
    parser_opt.add_argument("--max-attempts", type=int, default=3, help="Maximum training attempts")
    parser_opt.add_argument("--dataset", type=str, default="dataset", help="Base dataset directory")
    parser_opt.add_argument("--input-dir", type=str, default=None, help="Path to folder of real images")

    args = parser.parse_args()

    if args.command == "generate-dataset":
        handle_generate(args)
    elif args.command == "embed":
        handle_embed(args)
    elif args.command == "extract":
        handle_extract(args)
    elif args.command == "detect":
        handle_detect(args)
    elif args.command == "train":
        handle_train(args)
    elif args.command == "optimize":
        handle_optimize(args)

if __name__ == "__main__":
    main()
