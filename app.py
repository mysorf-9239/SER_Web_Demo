from flask import Flask, request, jsonify, render_template
import os
import librosa
import tempfile
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load model 1 lần duy nhất khi server khởi động
model = tf.keras.models.load_model('emotion_model.keras')

# Các nhãn cảm xúc tương ứng output của model
labels = ['neutral', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise', 'calm']

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)  # (40, time_steps)
    mfcc = mfcc.T  # (time_steps, 40)

    # Cắt hoặc pad time_steps = 155
    if mfcc.shape[0] < 155:
        pad_width = 155 - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:155, :]

    # Giữ lại 1 channel duy nhất như model yêu cầu
    mfcc = mfcc[:, 0:1]  # shape: (155, 1)
    mfcc = np.expand_dims(mfcc, axis=0)  # shape: (1, 155, 1)
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

    # Lưu tạm file vào tempfile
    with tempfile.NamedTemporaryFile(delete=True) as tmp:
        file.save(tmp.name)
        try:
            features = extract_features(tmp.name)
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)