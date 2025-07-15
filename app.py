from flask import Flask, request, jsonify, render_template
import numpy as np
import nibabel as nib
import cv2
import tensorflow as tf
from io import BytesIO

app = Flask(__name__)

# Carrega o modelo (coloque seu modelo na pasta static/model/)
model = tf.keras.models.load_model('static/model/model.h5', 
    custom_objects={
        'CustomCategoricalCrossentropy': CustomCategoricalCrossentropy,
        'dice_coef': dice_coef,
        # Adicione todas as métricas customizadas aqui
    })

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

    # Processa o NIfTI como no seu Colab
    nii_img = nib.load(file.stream)
    volume = nii_img.get_fdata()
    
    # Pré-processamento (igual ao seu código)
    X = np.zeros((VOLUME_SLICES, IMG_SIZE, IMG_SIZE, 2))
    for j in range(VOLUME_SLICES):
        X[j,:,:,0] = cv2.resize(volume[:,:,j+VOLUME_START_AT], (IMG_SIZE, IMG_SIZE))
    
    # Normalização e predição
    X = X / np.max(X)
    pred = model.predict(X[np.newaxis, ...])[0]  # Adiciona dimensão de batch

    # Salva a predição como imagem PNG
    output_img = (pred[60,:,:,1:4] * 255).astype('uint8')  # Fatia 60, classes 1-3
    _, img_encoded = cv2.imencode('.png', cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
    return img_encoded.tobytes()

if __name__ == '__main__':
    app.run(debug=True)
