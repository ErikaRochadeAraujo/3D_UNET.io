let model;
let originalVolume;

// Carregar o modelo quando a página abrir
async function loadModel() {
    model = await tf.loadLayersModel('assets/model/model.json');
    console.log("Modelo carregado com sucesso!");
}

// Processar arquivo NIfTI (simplificado)
async function processNIfTI(file) {
    return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = function(e) {
            // Aqui você precisaria de uma biblioteca como niimath.js para processar NIfTI
            // Esta é uma versão simplificada para demonstração
            const arrayBuffer = e.target.result;
            // Simulação: extrair uma fatia do volume
            const sliceData = extractSliceFromNIfTI(arrayBuffer, 60);
            resolve(sliceData);
        };
        reader.readAsArrayBuffer(file);
    });
}

// Função para exibir imagem no canvas
function displayImage(canvasId, imageData, width, height) {
    const canvas = document.getElementById(canvasId);
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    
    // Criar ImageData
    const imageDataObj = new ImageData(new Uint8ClampedArray(imageData), width, height);
    ctx.putImageData(imageDataObj, 0, 0);
}

// Predição principal
document.getElementById('predict-btn').addEventListener('click', async () => {
    const fileInput = document.getElementById('mri-upload');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Por favor, selecione um arquivo NIfTI (.nii ou .nii.gz)');
        return;
    }
    
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('slice-control').classList.remove('hidden');
    
    try {
        // Processar arquivo (versão simplificada)
        const sliceData = await processNIfTI(file);
        originalVolume = sliceData.volume;
        
        // Exibir fatia inicial
        updateSliceDisplay(60);
        
    } catch (error) {
        console.error('Erro:', error);
        alert('Ocorreu um erro ao processar a imagem: ' + error.message);
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
});

// Controle de fatias
document.getElementById('slice-number').addEventListener('input', function() {
    const sliceNum = parseInt(this.value);
    document.getElementById('slice-value').textContent = sliceNum;
    updateSliceDisplay(sliceNum);
});

async function updateSliceDisplay(sliceNum) {
    if (!originalVolume) return;
    
    // Extrair fatia do volume (simulação)
    const slice = extractSlice(originalVolume, sliceNum);
    
    // Exibir original
    displayImage('original-image', slice.gray, 128, 128);
    
    // Pré-processamento para o modelo
    const inputTensor = tf.tensor4d([slice.gray], [1, 128, 128, 1]);
    
    // Fazer predição
    const prediction = model.predict(inputTensor);
    const predictionData = await prediction.array();
    
    // Processar resultado (exemplo simplificado)
    const combinedPrediction = combinePredictions(predictionData[0]);
    
    // Exibir predição
    displayImage('prediction-result', combinedPrediction, 128, 128);
}

// Carregar o modelo quando a página carregar
window.onload = loadModel;