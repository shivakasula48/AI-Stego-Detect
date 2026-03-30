import os
import sys
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

logger = logging.getLogger(__name__)

try:
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from utils.preprocessing import load_and_preprocess_image
except ImportError as e:
    sys.exit(f"Error: {e}\nPlease install dependencies using pip install -r requirements.txt inside a virtual environment")


def predict_image(image_path, model_path="ai_model/stego_detector.h5"):
    """
    Loads the trained CNN model and predicts whether the image is clean or stego.
    Returns: (label: str, confidence: float) or (None, None) on failure.
    """
    if not os.path.exists(model_path):
        logger.error("Model not found. Please train the model first using: python cli.py train")
        return None, None

    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        return None, None

    try:
        model = load_model(model_path)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None, None

    img = load_and_preprocess_image(image_path, target_size=(128, 128))
    if img is None:
        logger.error("Could not preprocess image.")
        return None, None

    try:
        img_batch = np.expand_dims(img, axis=0)
        raw_pred = model.predict(img_batch, verbose=0)[0][0]
        prediction = float(raw_pred)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None, None

    # Confidence calculation: Sigmoid outputs [0, 1]
    # 0 = Clean, 1 = Stego
    if prediction < 0.5:
        label = "Clean Image"
        confidence = (1.0 - prediction) * 100
    else:
        label = "Stego Image"
        confidence = prediction * 100

    logger.info(f"Analysis Complete: [Raw: {prediction:.6f}] [Label: {label}] [Conf: {confidence:.2f}%]")
    return label, confidence


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        logger.error("Usage: python ai_model/predict.py <image_path>")
