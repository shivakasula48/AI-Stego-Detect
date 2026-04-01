
# AI-Based Encrypted Image Steganography Detection System

## Overview
This project provides a robust system for hiding encrypted messages inside images using advanced frequency-domain steganography (DCT), and detecting such stego images using a custom-trained Convolutional Neural Network (CNN). It includes both a CLI and a Flask web interface.

## Features
- AES encryption (CBC mode) for message confidentiality
- Lossless DCT steganography for hiding encrypted data in PNG images
- Extraction and decryption of hidden messages
- Dataset generator for AI-based steganalysis
- CNN model for detecting stego images
- CLI and Flask web UI

## Requirements
- Python 3.7+
- All dependencies are managed via `requirements.txt`

## Installation

```bash
# 1. Create a virtual environment
python -m venv venv

# 2. Activate the virtual environment
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# 3. Install required packages
pip install -r requirements.txt
```

## Usage

All functionality is provided via the `cli.py` entrypoint or the Flask web UI (`app.py`).

### 1. Generate Stego Dataset
Generates synthetic/clean images and automatically embeds secret data into them to prepare a dataset for AI training.
```bash
python cli.py generate-dataset --input dataset/clean/ --output dataset/stego/ --num-images 50
```

### 2. Encrypt and Embed Message
Injects a secret message string into an image losslessly without corrupting any underlying float definitions during compression.
```bash
python cli.py embed --input image.png --message "Super Secret" --output stego.png
```

### 3. Extract and Decrypt Message
Retrieves the underlying file bits and decrypts the original AES-CBC string utilizing your keys.
```bash
python cli.py extract --input stego.png
```

### 4. AI Detection (CNN)
Run the trained CNN to classify any image as **Stego** or **Clean**:
```bash
python cli.py detect --input image.png
```
Expected output:
```
[RESULT] Stego Image (Hidden Data Detected) | Confidence: 92.34%
[RESULT] Clean Image (No Hidden Data)       | Confidence: 88.10%
```

### 5. Train the CNN Model
Before detecting, first generate a dataset and then run training:
```bash
python cli.py train
# or specify a custom output path:
python cli.py train --output ai_model/stego_detector.h5
```

### 6. Auto-Optimize (One Command)
Automatically generates a dataset, trains the CNN, evaluates accuracy, and retries up to 3 times until the target validation accuracy is met:
```bash
python cli.py optimize
# Customize:
python cli.py optimize --num-images 400 --target 75.0 --max-attempts 3
```
Output includes:
```
Best Validation Accuracy:  XX.XX%
Clean Detection Accuracy:  XX% (X/5)
Stego Detection Accuracy:  XX% (X/5)
STATUS: ✅ TARGET MET / ⚠️ TARGET NOT MET
```

### 7. Flask Web UI
To launch the web interface:
```bash
python app.py
# Then open http://localhost:5000 in your browser
```
The web UI allows you to upload images, embed/extract messages, and run AI detection interactively.

## Project Structure
```
project/
├── encryption/
│   └── aes.py
├── steganography/
│   ├── embed.py
│   └── extract.py
├── dataset/
│   ├── train/
│   │   ├── clean/
│   │   └── stego/
│   ├── val/
│   │   ├── clean/
│   │   └── stego/
│   ├── generate_stego.py
│   └── image_utils.py
├── ai_model/
│   ├── train.py
│   ├── predict.py
│   ├── stego_detector.h5   <-- Trained model weights
│   ├── accuracy_plot.png
│   └── confusion_matrix.png
├── utils/
│   └── preprocessing.py
├── cli.py                   <-- Application Entry Point
├── app.py                   <-- Flask Web UI
├── requirements.txt
├── README.md
├── static/
│   ├── uploads/
│   └── outputs/
├── templates/
│   └── index.html
├── images/
└── ...
```

## Notes
- Always operate from the `venv`.
- Only PNG images are guaranteed against lossy pipeline corruptions!
- For best model accuracy, generate at least 400 image pairs using the `optimize` command.
- The CNN expects 128x128 images. Images are automatically resized if needed.
- All cryptographic operations use AES in CBC mode for strong security.
- The web UI and CLI share the same backend logic for consistency.

## License
This project is provided for educational and research purposes. Please review the LICENSE file for details.

## Contact
For questions, issues, or contributions, please open an issue or pull request on the GitHub repository.
=======
 Only PNG images are guaranteed against lossy pipeline corruptions!
- For best model accuracy, generate at least 400 image pairs using the `optimize` command.


## Freelance Project Notice

> 💼 **This is a freelance project** built and delivered for a client.
>
> We are **available for freelance work** — if you need a similar application or any custom software built, feel free to reach out!
>
> 📧 **Contact:** [shivakasula10@gmail.com](mailto:shivakasula10@gmail.com)
>
> We specialize in AI/ML applications, web development, facial recognition systems, and full-stack Python projects. Let's build something great together.
