document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const videoSide = document.getElementById('video');
    const videoTop = document.getElementById('video-top');
    const captureSideButton = document.getElementById('capture-side-button');
    const uploadSideButton = document.getElementById('upload-side-button');
    const fileInputSide = document.getElementById('file-input-side');
    const captureTopButton = document.getElementById('capture-top-button');
    const uploadTopButton = document.getElementById('upload-top-button');
    const fileInputTop = document.getElementById('file-input-top');
    const captureSection = document.getElementById('capture-section');
    const topSection = document.getElementById('top-section');
    const resultSection = document.getElementById('result-section');
    const chaiTspEl = document.getElementById('chai-tsp');
    const bubbleCountEl = document.getElementById('bubble-count');
    const sidePreviewEl = document.getElementById('side-preview');
    const topPreviewEl = document.getElementById('top-preview');
    const loadingEl = document.getElementById('loading');
    const toastEl = document.getElementById('toast');
    const analyzeAgainBtn = document.getElementById('analyze-again');
    const step1Indicator = document.getElementById('step1');
    const step2Indicator = document.getElementById('step2');

    let sideViewImage = null;
    let topViewImage = null;
    let showToastTimeout = null;

    // Toast helper
    function showToast(message, duration = 3000) {
        toastEl.textContent = message;
        toastEl.classList.add('show');
        
        clearTimeout(showToastTimeout);
        showToastTimeout = setTimeout(() => {
            toastEl.classList.remove('show');
        }, duration);
    }

    // Start camera for side view
    async function startSideCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { 
                    facingMode: 'environment',
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                } 
            });
            videoSide.srcObject = stream;
            captureSideButton.style.display = 'inline-block';
            uploadSideButton.style.display = 'inline-block';
        } catch (err) {
            console.error("Camera error:", err);
            captureSideButton.style.display = 'none';
            showToast("Camera not available. Please upload a photo.");
        }
    }

    // Start camera for top view
    async function startTopCamera() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: true,
                width: { ideal: 1280 },
                height: { ideal: 720 }
            });
            videoTop.srcObject = stream;
            captureTopButton.style.display = 'inline-block';
            uploadTopButton.style.display = 'inline-block';
        } catch (err) {
            console.error("Camera error:", err);
            captureTopButton.style.display = 'none';
            showToast("Camera not available. Please upload a photo.");
        }
    }

    // Capture image from camera
    function captureImage(videoElement) {
        const canvas = document.createElement('canvas');
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
        return canvas.toDataURL('image/jpeg', 0.9);
    }

    // Handle side view capture/upload
    captureSideButton.addEventListener('click', () => {
        sideViewImage = captureImage(videoSide);
        proceedToTopView();
    });
    
    uploadSideButton.addEventListener('click', () => fileInputSide.click());
    
    fileInputSide.addEventListener('change', () => {
        const file = fileInputSide.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            sideViewImage = e.target.result;
            proceedToTopView();
        };
        reader.readAsDataURL(file);
    });

    // Handle top view capture/upload
    captureTopButton.addEventListener('click', () => {
        topViewImage = captureImage(videoTop);
        analyzeChai();
    });
    
    uploadTopButton.addEventListener('click', () => fileInputTop.click());
    
    fileInputTop.addEventListener('change', () => {
        const file = fileInputTop.files[0];
        if (!file) return;
        
        const reader = new FileReader();
        reader.onload = (e) => {
            topViewImage = e.target.result;
            analyzeChai();
        };
        reader.readAsDataURL(file);
    });

    // Proceed to top view capture
    function proceedToTopView() {
        // Stop side camera
        if (videoSide.srcObject) {
            videoSide.srcObject.getTracks().forEach(track => track.stop());
        }
        
        // Update UI
        captureSection.style.display = 'none';
        topSection.style.display = 'block';
        step1Indicator.classList.remove('step-active');
        step1Indicator.classList.add('step-completed');
        step2Indicator.classList.remove('step-inactive');
        step2Indicator.classList.add('step-active');
        
        // Start top camera
        startTopCamera();
    }

    // Analyze both images
    async function analyzeChai() {
        if (!sideViewImage || !topViewImage) {
            showToast("Please capture both side and top views");
            return;
        }
        
        // Stop top camera
        if (videoTop.srcObject) {
            videoTop.srcObject.getTracks().forEach(track => track.stop());
        }
        
        topSection.style.display = 'none';
        loadingEl.style.display = 'block';
        
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    side_view: sideViewImage,
                    top_view: topViewImage
                })
            });
            
            if (!response.ok) {
                throw new Error(await response.text());
            }
            
            const data = await response.json();
            showResults(data);
        } catch (err) {
            console.error("Analysis error:", err);
            showToast("Analysis failed. Please try again.");
            resetUI();
        } finally {
            loadingEl.style.display = 'none';
        }
    }

    // Display results
    function showResults(data) {
        // Display simplified results in UI
        chaiTspEl.textContent = data.chai_teaspoons.toFixed(1);
        bubbleCountEl.textContent = data.bubble_count;
        
        // Show annotated images if available
        if (data.annotated_side) {
            sidePreviewEl.src = data.annotated_side;
        }
        
        if (data.annotated_top) {
            topPreviewEl.src = data.annotated_top;
        }
        
        // Log detailed results to console
        console.log("Detailed Chai Analysis:", {
            chaiHeight: data.chai_height,
            frothHeight: data.froth_height,
            ratio: data.ratio,
            chaiML: data.chai_ml,
            frothML: data.froth_ml,
            roast: data.roast
        });
        
        resultSection.style.display = 'block';
    }

    // Reset UI
    function resetUI() {
        captureSection.style.display = 'block';
        topSection.style.display = 'none';
        resultSection.style.display = 'none';
        
        step1Indicator.classList.add('step-active');
        step1Indicator.classList.remove('step-completed');
        step2Indicator.classList.add('step-inactive');
        step2Indicator.classList.remove('step-active');
        
        sideViewImage = null;
        topViewImage = null;
        fileInputSide.value = '';
        fileInputTop.value = '';
        
        startSideCamera();
    }

    // Event listeners
    analyzeAgainBtn.addEventListener('click', resetUI);

    // Initialize
    startSideCamera();
});