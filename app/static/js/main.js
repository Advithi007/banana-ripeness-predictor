/**
 * Days to Banana Death — Frontend Logic
 *
 * Handles:
 * - Drag-and-drop image upload
 * - File input selection
 * - Image preview
 * - API call to /predict
 * - Animated result rendering
 * - Ripeness timeline visualization
 * - Background particles
 */

document.addEventListener('DOMContentLoaded', () => {
    // === DOM Elements ===
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('previewContainer');
    const previewImage = document.getElementById('previewImage');
    const btnRemove = document.getElementById('btnRemove');
    const btnPredict = document.getElementById('btnPredict');
    const btnText = document.querySelector('.btn-text');
    const btnLoader = document.getElementById('btnLoader');
    const resultSection = document.getElementById('resultSection');
    const uploadSection = document.getElementById('uploadSection');
    const heroSection = document.getElementById('heroSection');

    // Result elements
    const daysNumber = document.getElementById('daysNumber');
    const resultEmoji = document.getElementById('resultEmoji');
    const timelineFill = document.getElementById('timelineFill');
    const timelineMarker = document.getElementById('timelineMarker');
    const markerLabel = document.getElementById('markerLabel');
    const modelBadge = document.getElementById('modelBadge');
    const btnTryAgain = document.getElementById('btnTryAgain');

    let selectedFile = null;

    // === Background Particles ===
    (function initParticles() {
        const container = document.getElementById('bgParticles');
        if (!container) return;
        for (let i = 0; i < 20; i++) {
            const particle = document.createElement('div');
            particle.className = 'particle';
            particle.style.left = Math.random() * 100 + '%';
            particle.style.animationDelay = Math.random() * 20 + 's';
            particle.style.animationDuration = (15 + Math.random() * 20) + 's';
            container.appendChild(particle);
        }
    })();

    // === Drag & Drop ===
    uploadZone.addEventListener('click', () => fileInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('drag-over');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('drag-over');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('drag-over');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });

    // === File Input ===
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // === Handle File Selection ===
    function handleFile(file) {
        // Validate file type
        const validTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp'];
        if (!validTypes.includes(file.type)) {
            showError('Please upload a valid image (JPG, PNG, or WebP).');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showError('Image too large. Maximum size is 10MB.');
            return;
        }

        selectedFile = file;
        showPreview(file);
    }

    // === Image Preview ===
    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadZone.style.display = 'none';
            previewContainer.style.display = 'block';
            btnPredict.style.display = 'block';

            // Hide result if showing
            resultSection.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // === Remove Image ===
    btnRemove.addEventListener('click', () => {
        resetUpload();
    });

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewImage.src = '';
        uploadZone.style.display = 'flex';
        previewContainer.style.display = 'none';
        btnPredict.style.display = 'none';
        resultSection.style.display = 'none';

        // Clear any error messages
        const errors = document.querySelectorAll('.error-message');
        errors.forEach(el => el.remove());
    }

    // === Predict Button ===
    btnPredict.addEventListener('click', async () => {
        if (!selectedFile) return;

        // Show loading state
        btnPredict.disabled = true;
        btnText.style.display = 'none';
        btnLoader.style.display = 'flex';

        // Clear previous errors
        const errors = document.querySelectorAll('.error-message');
        errors.forEach(el => el.remove());

        try {
            const formData = new FormData();
            formData.append('file', selectedFile);

            const response = await fetch('/predict', {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok || data.error) {
                throw new Error(data.error || 'Prediction failed');
            }

            showResult(data);

        } catch (error) {
            showError(error.message);
        } finally {
            btnPredict.disabled = false;
            btnText.style.display = 'flex';
            btnLoader.style.display = 'none';
        }
    });

    // === Show Result ===
    function showResult(data) {
        const days = data.days_remaining;
        const daysRounded = data.days_remaining_rounded;

        // Scroll to result
        resultSection.style.display = 'block';

        // Animate the days number counting up
        animateCounter(daysNumber, days, 1200);

        // Set emoji based on ripeness
        resultEmoji.textContent = getRipenessEmoji(days);

        // Animate timeline
        // Assume max lifespan ~8 days for visualization
        // Position is inverted: more days remaining = less ripe
        const maxDays = 8;
        const progress = Math.max(0, Math.min(100, ((maxDays - days) / maxDays) * 100));

        setTimeout(() => {
            timelineFill.style.width = progress + '%';
            timelineMarker.style.left = progress + '%';
            markerLabel.textContent = getRipenessStage(days);
        }, 300);

        // Model info
        modelBadge.textContent = `Model: ${data.model_used}`;

        // Smooth scroll to result
        setTimeout(() => {
            resultSection.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }, 200);
    }

    // === Animate Counter ===
    function animateCounter(element, target, duration) {
        const start = 0;
        const startTime = performance.now();

        function update(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = start + (target - start) * eased;

            element.textContent = current.toFixed(1);

            if (progress < 1) {
                requestAnimationFrame(update);
            }
        }

        requestAnimationFrame(update);
    }

    // === Get Ripeness Emoji ===
    function getRipenessEmoji(days) {
        if (days >= 5) return '🟢';   // Green/fresh
        if (days >= 3) return '🍌';   // Yellow/ripe
        if (days >= 1) return '🟡';   // Spotted/very ripe
        return '🟤';                   // Overripe
    }

    // === Get Ripeness Stage Label ===
    function getRipenessStage(days) {
        if (days >= 5) return 'Still green — plenty of time';
        if (days >= 3) return 'Ripe — good to eat';
        if (days >= 1) return 'Getting spotted — eat soon!';
        return 'Almost overripe — eat now!';
    }

    // === Show Error ===
    function showError(message) {
        // Remove existing errors
        const existing = document.querySelectorAll('.error-message');
        existing.forEach(el => el.remove());

        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;

        // Insert after predict button
        btnPredict.parentNode.insertBefore(errorDiv, btnPredict.nextSibling);

        // Auto-remove after 6 seconds
        setTimeout(() => {
            if (errorDiv.parentNode) {
                errorDiv.style.opacity = '0';
                errorDiv.style.transform = 'translateY(-10px)';
                setTimeout(() => errorDiv.remove(), 300);
            }
        }, 6000);
    }

    // === Try Again ===
    btnTryAgain.addEventListener('click', () => {
        resetUpload();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });
});
