let imagesData = []; // { file, objectUrl, trueLabel, predLabel, confidence }
let currentFilter = 'all';
let isRunning = false;

// DOM Elements
const fileInput = document.getElementById('folder-upload');
const fileCountText = document.getElementById('file-count');
const autoLabelBtn = document.getElementById('auto-label-btn');
const runInferenceBtn = document.getElementById('run-inference-btn');
const progressBar = document.getElementById('progress-bar');
const gallery = document.getElementById('image-gallery');

// Metrics Elements
const elAcc = document.getElementById('metric-accuracy');
const elPrec = document.getElementById('metric-precision');
const elRec = document.getElementById('metric-recall');
const elTP = document.getElementById('cm-tp');
const elTN = document.getElementById('cm-tn');
const elFP = document.getElementById('cm-fp');
const elFN = document.getElementById('cm-fn');

fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files).filter(f => f.type.startsWith('image/'));
    if (files.length === 0) return;

    // Reset state
    imagesData = files.map(file => ({
        file: file,
        objectUrl: URL.createObjectURL(file),
        trueLabel: null,
        predLabel: null,
        confidence: null
    }));

    fileCountText.textContent = `${imagesData.length} images loaded`;
    autoLabelBtn.disabled = false;
    runInferenceBtn.disabled = false;
    
    renderGallery();
    resetMetrics();
});

autoLabelBtn.addEventListener('click', () => {
    let labeled = 0;
    imagesData.forEach(data => {
        // Kaggle dataset format: 1_xxx.png or 0_xxx.png
        const match = data.file.name.match(/^([01])_/);
        if (match) {
            data.trueLabel = parseInt(match[1]);
            labeled++;
        }
    });
    alert(`Auto-labeled ${labeled} images based on filename.`);
    renderGallery();
});

runInferenceBtn.addEventListener('click', async () => {
    if (isRunning || imagesData.length === 0) return;
    isRunning = true;
    runInferenceBtn.disabled = true;
    runInferenceBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
    
    resetMetrics();
    progressBar.style.width = '0%';
    
    // Batch processing limits
    const concurrencyLimit = 5;
    let index = 0;
    
    const processNext = async () => {
        if (index >= imagesData.length) return;
        const currentIndex = index++;
        const data = imagesData[currentIndex];
        
        const formData = new FormData();
        formData.append('file', data.file);
        
        try {
            const res = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                body: formData
            });
            const result = await res.json();
            
            if (result.error) {
                console.error("Inference Error:", result.error);
            } else {
                data.predLabel = result.prediction;
                // Fallback just in case backend hasn't restarted and still returns 'confidence'
                if (result.prob_ship !== undefined) {
                    data.prob_ship = result.prob_ship;
                    data.prob_no_ship = result.prob_no_ship;
                } else {
                    data.prob_ship = result.prediction === 1 ? result.confidence : 1 - result.confidence;
                    data.prob_no_ship = 1 - data.prob_ship;
                }
                updateMetrics();
                updateGalleryItem(currentIndex);
            }
        } catch (err) {
            console.error("Network Error:", err);
        }
        
        // Update progress
        const progress = Math.min(100, Math.round(((currentIndex + 1) / imagesData.length) * 100));
        progressBar.style.width = `${progress}%`;
        
        await processNext();
    };
    
    // Start concurrent workers
    const workers = [];
    for(let i=0; i < concurrencyLimit; i++){
        workers.push(processNext());
    }
    await Promise.all(workers);
    
    isRunning = false;
    runInferenceBtn.disabled = false;
    runInferenceBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> Start Live Inference';
});

// Filtering
document.querySelectorAll('.filter-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        currentFilter = e.target.dataset.filter;
        renderGallery();
    });
});

function renderGallery() {
    gallery.innerHTML = '';
    if (imagesData.length === 0) {
        gallery.innerHTML = `
            <div class="empty-state">
                <i class="fa-regular fa-image"></i>
                <p>Upload a folder to begin.</p>
            </div>`;
        return;
    }
    
    const fragment = document.createDocumentFragment();
    imagesData.forEach((data, index) => {
        if (currentFilter === 'errors' && data.trueLabel !== null && data.predLabel !== null) {
            if (data.trueLabel === data.predLabel) return; // Skip correct ones
        }
        
        const item = document.createElement('div');
        item.className = 'img-item';
        item.id = `img-item-${index}`;
        
        // Status classes
        if (data.trueLabel !== null && data.predLabel !== null) {
            item.classList.add(data.trueLabel === data.predLabel ? 'correct' : 'incorrect');
        }
        
        const trueText = data.trueLabel !== null ? (data.trueLabel ? 'Ship' : 'No') : '?';
        const predText = data.predLabel !== null ? (data.predLabel ? 'Ship' : 'No') : '-';
        const matchClass = (data.trueLabel !== null && data.predLabel !== null) 
            ? (data.trueLabel === data.predLabel ? 'pred-match' : 'pred-mismatch') 
            : '';
            
        let probHtml = '';
        if (data.prob_ship !== undefined && data.prob_ship !== null) {
            probHtml = `
                <div class="prob-container">
                    <div class="prob-row"><span>🚢 Barco</span> <span>${(data.prob_ship*100).toFixed(1)}%</span></div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill fill-ship" style="width: ${data.prob_ship*100}%"></div></div>
                    <div class="prob-row"><span>🌊 No Barco</span> <span>${(data.prob_no_ship*100).toFixed(1)}%</span></div>
                    <div class="prob-bar-bg"><div class="prob-bar-fill fill-noship" style="width: ${data.prob_no_ship*100}%"></div></div>
                </div>
            `;
        }
            
        item.innerHTML = `
            <div class="img-wrapper">
                <img src="${data.objectUrl}" alt="chip" loading="lazy" />
                <div class="img-overlay">
                    <span class="label-true">True: ${trueText}</span>
                    <span class="label-pred ${matchClass}">Pred: ${predText}</span>
                </div>
            </div>
            
            ${probHtml}
            
            <div class="label-controls">
                <button class="label-btn btn-ship" onclick="setLabel(${index}, 1)"><i class="fa-solid fa-ship"></i> Barco</button>
                <button class="label-btn btn-noship" onclick="setLabel(${index}, 0)"><i class="fa-solid fa-water"></i> No Barco</button>
            </div>
        `;
        fragment.appendChild(item);
    });
    
    gallery.appendChild(fragment);
}

function updateGalleryItem(index) {
    const data = imagesData[index];
    const item = document.getElementById(`img-item-${index}`);
    if (!item) return; // Filtered out
    
    if (data.trueLabel !== null && data.predLabel !== null) {
        item.classList.remove('correct', 'incorrect');
        item.classList.add(data.trueLabel === data.predLabel ? 'correct' : 'incorrect');
    }
    
    const trueText = data.trueLabel !== null ? (data.trueLabel ? 'Ship' : 'No') : '?';
    const predText = data.predLabel !== null ? (data.predLabel ? 'Ship' : 'No') : '-';
    const matchClass = (data.trueLabel !== null && data.predLabel !== null) 
        ? (data.trueLabel === data.predLabel ? 'pred-match' : 'pred-mismatch') 
        : '';
        
    let probHtml = '';
    if (data.prob_ship !== undefined && data.prob_ship !== null) {
        probHtml = `
            <div class="prob-container">
                <div class="prob-row"><span>🚢 Barco</span> <span>${(data.prob_ship*100).toFixed(1)}%</span></div>
                <div class="prob-bar-bg"><div class="prob-bar-fill fill-ship" style="width: ${data.prob_ship*100}%"></div></div>
                <div class="prob-row"><span>🌊 No Barco</span> <span>${(data.prob_no_ship*100).toFixed(1)}%</span></div>
                <div class="prob-bar-bg"><div class="prob-bar-fill fill-noship" style="width: ${data.prob_no_ship*100}%"></div></div>
            </div>
        `;
    }
        
    item.innerHTML = `
        <div class="img-wrapper">
            <img src="${data.objectUrl}" alt="chip" loading="lazy" />
            <div class="img-overlay">
                <span class="label-true">True: ${trueText}</span>
                <span class="label-pred ${matchClass}">Pred: ${predText}</span>
            </div>
        </div>
        
        ${probHtml}
        
        <div class="label-controls">
            <button class="label-btn btn-ship" onclick="setLabel(${index}, 1)"><i class="fa-solid fa-ship"></i> Barco</button>
            <button class="label-btn btn-noship" onclick="setLabel(${index}, 0)"><i class="fa-solid fa-water"></i> No Barco</button>
        </div>
    `;
}

window.setLabel = function(index, label) {
    imagesData[index].trueLabel = label;
    updateGalleryItem(index);
    updateMetrics();
};

function resetMetrics() {
    elAcc.textContent = '0.00%';
    elPrec.textContent = '0.00%';
    elRec.textContent = '0.00%';
    elTP.textContent = '0';
    elTN.textContent = '0';
    elFP.textContent = '0';
    elFN.textContent = '0';
}

function updateMetrics() {
    let tp = 0, tn = 0, fp = 0, fn = 0;
    
    imagesData.forEach(data => {
        if (data.trueLabel !== null && data.predLabel !== null) {
            if (data.trueLabel === 1 && data.predLabel === 1) tp++;
            else if (data.trueLabel === 0 && data.predLabel === 0) tn++;
            else if (data.trueLabel === 0 && data.predLabel === 1) fp++;
            else if (data.trueLabel === 1 && data.predLabel === 0) fn++;
        }
    });
    
    const total = tp + tn + fp + fn;
    if (total === 0) return;
    
    const acc = (tp + tn) / total;
    const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
    const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
    
    elAcc.textContent = (acc * 100).toFixed(2) + '%';
    elPrec.textContent = (prec * 100).toFixed(2) + '%';
    elRec.textContent = (rec * 100).toFixed(2) + '%';
    
    elTP.textContent = tp;
    elTN.textContent = tn;
    elFP.textContent = fp;
    elFN.textContent = fn;
    
    // Penalization visual logic (>98%)
    if (acc < 0.98) {
        elAcc.style.color = 'var(--danger)';
    } else {
        elAcc.style.color = 'var(--success)';
    }
}
