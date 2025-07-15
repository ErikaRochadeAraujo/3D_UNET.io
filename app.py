from flask import Flask, request, jsonify, render_template
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
from io import BytesIO
import tensorflowjs as tfjs  # Importação necessária

app = Flask(__name__)

# Configurações do modelo (ajuste conforme seu caso)
VOLUME_SLICES = 100
VOLUME_START_AT = 22
IMG_SIZE = 128

# Carrega o modelo TensorFlow.js
model = tfjs.converters.load_keras_model('static/model/model.json')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Arquivo inválido'}), 400

    try:
        # Processa o NIfTI
        nii_img = nib.load(file.stream)
        volume = nii_img.get_fdata()
        
        # Pré-processamento
        X = np.zeros((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
        for j in range(VOLUME_SLICES):
            X[j,:,:,0] = cv2.resize(volume[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
        
        # Normalização
        X = X / np.max(X)
        
        # Predição (adiciona dimensão de batch)
        pred = model.predict(X[np.newaxis, ...])[0]

        # Processa a saída (fatia 60 como exemplo)
        output_img = (pred[60,:,:,1:4] * 255).astype('uint8')
        _, img_encoded = cv2.imencode('.png', cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        
        return img_encoded.tobytes()

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
