// Elements
const video = document.getElementById('video');
const captureButton = document.getElementById('capture-button');
const uploadSideButton = document.getElementById('upload-side-button');
const fileInputSide = document.getElementById('file-input-side');

const captureTopButton = document.getElementById('capture-top-button');
const uploadTopButton = document.getElementById('upload-top-button');
const fileInputTop = document.getElementById('file-input-top');

const resultSection = document.getElementById('result-section');
const roastP = document.getElementById('roast');
const chaiPctEl = document.getElementById('chai-pct');
const frothPctEl = document.getElementById('froth-pct');
const ratioPctEl = document.getElementById('ratio-pct');
const bubbleCountEl = document.getElementById('bubble-count');
const scoreCircle = document.getElementById('score-circle');
const scoreText = document.getElementById('score-text');

const loadingEl = document.getElementById('loading');
const toastEl = document.getElementById('toast');
const analyzeAgainBtn = document.getElementById('analyze-again');

let sideViewImage = null;
let topViewImage = null;

// Utility: show toast
function showToast(msg, timeout = 3000) {
    toastEl.textContent = msg;
    toastEl.style.display = 'block';
    clearTimeout(showToast._t);
    showToast._t = setTimeout(() => toastEl.style.display = 'none', timeout);
}

// --- Camera setup ---
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } });
        video.srcObject = stream;
        captureButton.style.display = 'inline-block';
        uploadSideButton.style.display = 'inline-block';
        captureTopButton.style.display = 'none';
        uploadTopButton.style.display = 'none';
    } catch (e) {
        console.warn('Camera not accessible, fallback to upload');
        captureButton.style.display = 'none';
        captureTopButton.style.display = 'none';
        uploadSideButton.style.display = 'inline-block';
        uploadTopButton.style.display = 'none';
        showToast('Camera not available, use upload.');
    }
}
startCamera();

// Canvas capture helper (maintain video aspect)
function captureFrameFromVideo() {
    const canvas = document.createElement('canvas');
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    canvas.width = vw;
    canvas.height = vh;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/png');
}

// --- Side view handling ---
captureButton.addEventListener('click', () => {
    sideViewImage = captureFrameFromVideo();
    // Switch UI to allow top view
    captureButton.style.display = 'none';
    uploadSideButton.style.display = 'none';
    captureTopButton.style.display = 'inline-block';
    uploadTopButton.style.display = 'inline-block';
    showToast('Side view captured. Now capture/upload top view.');
});

uploadSideButton.addEventListener('click', () => fileInputSide.click());
fileInputSide.onchange = () => {
    const file = fileInputSide.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        sideViewImage = reader.result;
        captureButton.style.display = 'none';
        uploadSideButton.style.display = 'none';
        captureTopButton.style.display = 'inline-block';
        uploadTopButton.style.display = 'inline-block';
        showToast('Side image uploaded. Now capture/upload top view.');
    };
    reader.readAsDataURL(file);
};

// --- Top view handling ---
captureTopButton.addEventListener('click', () => {
    if (!sideViewImage) { showToast('Please capture/upload side view first.'); return; }
    topViewImage = captureFrameFromVideo();
    sendImages();
});

uploadTopButton.addEventListener('click', () => fileInputTop.click());
fileInputTop.onchange = () => {
    const file = fileInputTop.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        topViewImage = reader.result;
        sendImages();
    };
    reader.readAsDataURL(file);
};

// --- Send images to backend ---
async function sendImages() {
    if (!sideViewImage || !topViewImage) {
        showToast('Both side and top images are required.');
        return;
    }

    // UI: show loading
    loadingEl.style.display = 'block';
    resultSection.style.display = 'none';

    const payload = {
        side_view: sideViewImage,
        top_view: topViewImage
    };

    try {
        const response = await fetch('/upload/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errText = await response.text();
            throw new Error(errText || 'Server error');
        }

        const data = await response.json();
        showResults(data);
    } catch (err) {
        console.error(err);
        showToast('Error analyzing images. Try again.');
    } finally {
        loadingEl.style.display = 'none';
    }
}

function clamp(n, min, max) { return Math.max(min, Math.min(max, n)); }
// ... (rest of the code stays same until showResults)

const chaiMlEl = document.getElementById('chai-ml');
const chaiTspEl = document.getElementById('chai-tsp');

function showResults(data) {
    resultSection.style.display = 'block';

    roastP.textContent = data.roast || 'No roast available.';

    const chaiPct = parseFloat(data.chai_height) || 0;
    const frothPct = parseFloat(data.froth_height) || 0;
    const ratioPct = parseFloat(data.ratio) || 0;
    const chaiMl = parseFloat(data.chai_ml) || 0;
    const chaiTsp = parseFloat(data.chai_teaspoons) || 0;
    const bubbleCount = Number.isInteger(data.bubble_count) ? data.bubble_count : Math.round(Number(data.bubble_count) || 0);

    chaiPctEl.textContent = `${chaiPct.toFixed(2)} %`;
    frothPctEl.textContent = `${frothPct.toFixed(2)} %`;
    ratioPctEl.textContent = `${ratioPct.toFixed(2)} %`;
    chaiMlEl.textContent = chaiMl.toFixed(2);
    chaiTspEl.textContent = chaiTsp.toFixed(2);
    bubbleCountEl.textContent = bubbleCount;

    const visualRating = clamp(ratioPct, 0, 100);
    scoreCircle.style.setProperty('--rating', visualRating.toString());
    scoreText.textContent = `${Math.round(visualRating)}%`;

    captureButton.style.display = 'inline-block';
    uploadSideButton.style.display = 'inline-block';
    captureTopButton.style.display = 'none';
    uploadTopButton.style.display = 'none';
    sideViewImage = null;
    topViewImage = null;
    showToast('Analysis complete — enjoy your roast!');
}
