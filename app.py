from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf  # ou torch, dependendo do seu modelo
import io
import os

app = Flask(__name__)

# Carregar o modelo
model = tf.keras.models.load_model('models/seu_modelo.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Nenhum arquivo enviado'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Nenhum arquivo selecionado'}), 400
    
    # Processar a imagem
    img = Image.open(io.BytesIO(file.read()))
    img = img.convert('RGB')  # Converter se necessário
    img = img.resize((256, 256))  # Ajustar conforme seu modelo espera
    
    # Pré-processamento
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Fazer a previsão
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction[0])
    
    # Mapear classes (ajuste conforme seu modelo)
    classes = {
        0: 'Sem tumor',
        1: 'Tumor detectado',
        # Adicione mais classes se necessário
    }
    
    result = {
        'prediction': classes[class_idx],
        'confidence': float(prediction[0][class_idx])
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)