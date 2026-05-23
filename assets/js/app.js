let model;

async function loadModel() {
    model = await tf.loadLayersModel('static/model/model.json');
    console.log("Modelo carregado!");
}

async function processNIfTI(file) {
    // Implemente usando https://github.com/rii-mango/NIfTI-Reader-JS
    // Ou converta para PNG no frontend
}

async function predict() {
    const file = document.getElementById('nii-upload').files[0];
    if (!file) return alert('Selecione um arquivo!');
    
    const tensor = await processNIfTI(file);
    const prediction = model.predict(tensor);
    
    // Visualização no canvas
    renderPrediction(prediction);
}

// Inicialização
loadModel();