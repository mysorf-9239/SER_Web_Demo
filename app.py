from flask import Flask, request, jsonify, render_template
import os
import gdown
import librosa
import numpy as np
import tensorflow as tf
import soundfile as sf
import io
import subprocess
import tempfile

app = Flask(__name__)

MODEL_PATH = 'emotion_model.keras'
MODEL_GDRIVE_URL = 'https://drive.google.com/uc?id=1SKMhP25iVSD_97thVB8rHB1jpNdknsVC'

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(MODEL_GDRIVE_URL, MODEL_PATH, quiet=False)
    else:
        print("Model already exists locally.")

download_model()
model = tf.keras.models.load_model(MODEL_PATH)

labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'calm']

def convert_webm_to_wav(webm_bytes):
    with tempfile.NamedTemporaryFile(suffix=".webm") as webm_file, \
         tempfile.NamedTemporaryFile(suffix=".wav") as wav_file:
        webm_file.write(webm_bytes)
        webm_file.flush()

        command = [
            'ffmpeg', '-y',  # overwrite output file if exists
            '-i', webm_file.name,
            '-ar', '22050',  # sample rate
            '-ac', '1',      # mono channel
            wav_file.name
        ]

        try:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            wav_file.seek(0)
            return wav_file.read()
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'ffmpeg conversion failed: {e.stderr.decode()}')

def extract_features(file_obj):
    # file_obj là BytesIO chứa file WAV
    y, sr = sf.read(file_obj)
    if len(y.shape) > 1:
        y = y.mean(axis=1)  # stereo -> mono

    y = y.astype(np.float32)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = mfcc.T

    if mfcc.shape[0] < 155:
        pad_width = 155 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:155, :]

    mfcc = mfcc[:, 0:1]
    mfcc = np.expand_dims(mfcc, axis=0)
    return mfcc.astype(np.float32)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'Không có file audio'}), 400

    file = request.files['audio_file']
    if file.filename == '':
        return jsonify({'error': 'Tên file không hợp lệ'}), 400

    try:
        audio_bytes_raw = file.read()

        # Nếu file .webm (định dạng ghi âm), convert sang wav trước
        if file.filename.lower().endswith('.webm'):
            audio_bytes = convert_webm_to_wav(audio_bytes_raw)
        else:
            audio_bytes = audio_bytes_raw

        audio_buffer = io.BytesIO(audio_bytes)
        features = extract_features(audio_buffer)

        preds = model.predict(features)
        predicted_index = np.argmax(preds)
        predicted_label = labels[predicted_index]
        confidence = float(preds[0][predicted_index])

        return jsonify({
            'label': predicted_label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # port = int(os.environ.get('PORT', 10000))
    # app.run(host='0.0.0.0', port=port)
    app.run(debug=True)