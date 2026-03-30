import os
import glob
import logging
from werkzeug.utils import secure_filename
from flask import Flask, request, render_template, jsonify

from steganography.embed import embed_data
from steganography.extract import extract_data
from encryption.aes import encrypt_message, decrypt_message
from dataset.generate_stego import generate_stego_dataset
from ai_model.predict import predict_image

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/embed', methods=['POST'])
def handle_embed():
    if 'image' not in request.files or 'message' not in request.form:
        return jsonify({'error': 'Missing image or message'}), 400

    file = request.files['image']
    message = request.form['message']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(input_path)

    enc_bytes = encrypt_message(message)
    if not enc_bytes:
        return jsonify({'error': 'AES encryption failed'}), 500

    out_filename = f"stego_{filename}"
    out_path = os.path.join(app.config['OUTPUT_FOLDER'], out_filename)

    success = embed_data(input_path, enc_bytes, out_path)
    if success:
        return jsonify({'success': True, 'output_url': f"/{out_path}"})
    else:
        return jsonify({'error': 'Embedding failed'}), 500


@app.route('/extract', methods=['POST'])
def handle_extract():
    if 'image' not in request.files:
        return jsonify({'error': 'Missing image'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"extracting_{filename}")
    file.save(input_path)

    try:
        enc_bytes = extract_data(input_path)
        if not enc_bytes:
            return jsonify({'error': 'Extraction returned no data'}), 500

        msg = decrypt_message(enc_bytes)
        if msg and not msg.startswith("[ERROR]"):
            return jsonify({'success': True, 'message': msg})
        else:
            return jsonify({'error': f"Decryption failed: {msg}"}), 500
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/detect', methods=['POST'])
def handle_detect():
    if 'image' not in request.files:
        return jsonify({'error': 'Missing image'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    input_path = os.path.join(app.config['UPLOAD_FOLDER'], f"detect_{filename}")
    file.save(input_path)

    try:
        label, confidence = predict_image(input_path)
        if label is None:
            return jsonify({'error': 'Detection failed. Ensure the model is trained first (python cli.py train).'}), 500

        is_stego = label == "Stego Image"
        return jsonify({
            'success': True,
            'label': label,
            'confidence': round(confidence, 2),
            'is_stego': is_stego,
            'preview_url': f"/{input_path}"
        })
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/generate', methods=['POST'])
def handle_generate():
    clean_dir = 'static/uploads/clean'
    stego_dir = 'static/outputs/stego'
    os.makedirs(clean_dir, exist_ok=True)
    os.makedirs(stego_dir, exist_ok=True)

    try:
        generate_stego_dataset(clean_dir=clean_dir, stego_dir=stego_dir, num_images=3)
        files = glob.glob(f"{stego_dir}/*.png")
        urls = [f"/{f}" for f in files]
        return jsonify({'success': True, 'images': urls})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
