document.addEventListener('DOMContentLoaded', () => {
    const webcamVideo = document.getElementById('webcam');
    const emotionCanvas = document.getElementById('emotionCanvas');
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const statusMessage = document.getElementById('statusMessage');
    const emotionDisplay = document.getElementById('emotionDisplay'); // This is the main emotion card
    const emotionDisplayText = emotionDisplay.querySelector('span'); // The span inside the card

    let stream = null;
    let canvasContext = emotionCanvas.getContext('2d');
    let videoWidth = 0;
    let videoHeight = 0;
    let detectionInterval = null;

    // IMPORTANT: Replace with your backend URL
    const BACKEND_URL = 'http://127.0.0.1:5000/predict_emotion';

    // --- Temporal Smoothing & Confidence Thresholding Settings ---
    const SMOOTHING_WINDOW_SIZE = 10; // Number of past frames to consider for smoothing
    const MIN_CONFIDENCE_THRESHOLD = 0.6; // Only show emotion if dominant emotion confidence is > 60%

    let predictionHistory = []; // Stores recent dominant emotion predictions and their scores
    let currentOverallEmotion = 'No Face Detected'; // To track the currently displayed emotion for animation triggers
    // --- End Settings ---

    // Define emotion colors for CANVAS DRAWING (BGR for OpenCV in backend, but here for drawing on JS canvas)
    // These are *not* directly related to Tailwind's classes but for the overlays.
    const EMOTION_CANVAS_COLORS = {
        'Angry': 'rgba(239, 68, 68, 0.8)',      // Tailwind red-500 equivalent
        'Disgust': 'rgba(34, 197, 94, 0.8)',    // Tailwind green-500
        'Fear': 'rgba(139, 92, 246, 0.8)',     // Tailwind violet-500
        'Happy': 'rgba(245, 158, 11, 0.8)',     // Tailwind amber-500
        'Sad': 'rgba(59, 130, 246, 0.8)',       // Tailwind blue-500
        'Surprise': 'rgba(6, 182, 212, 0.8)',   // Tailwind cyan-500
        'Neutral': 'rgba(100, 116, 139, 0.8)',  // Tailwind slate-500
        'Undetermined': 'rgba(148, 163, 184, 0.8)', // Tailwind gray-400
        'Analyzing...': 'rgba(251, 191, 36, 0.8)'   // Tailwind yellow-400
    };

    startButton.addEventListener('click', startCamera);
    stopButton.addEventListener('click', stopCamera);

    async function startCamera() {
        try {
            statusMessage.textContent = 'Requesting camera access...';
            statusMessage.classList.add('animate-fade-in-out'); // Start pulsing animation
            startButton.disabled = true;
            stopButton.disabled = false;

            stream = await navigator.mediaDevices.getUserMedia({ video: true });
            webcamVideo.srcObject = stream;

            webcamVideo.onloadedmetadata = () => {
                videoWidth = webcamVideo.videoWidth;
                videoHeight = webcamVideo.videoHeight;
                emotionCanvas.width = videoWidth;
                emotionCanvas.height = videoHeight;
                console.log(`Video dimensions: ${videoWidth}x${videoHeight}`);

                webcamVideo.play();
                statusMessage.textContent = 'Camera active. Detecting emotions...';
                statusMessage.classList.remove('animate-fade-in-out'); // Stop pulsing once active

                predictionHistory = [];
                currentOverallEmotion = 'No Face Detected'; // Reset current displayed emotion
                updateEmotionDisplay('Analyzing...'); // Initial state for emotion card

                detectionInterval = setInterval(sendFrameForDetection, 200);
            };

        } catch (err) {
            console.error('Error accessing camera:', err);
            statusMessage.textContent = `Error: ${err.message}. Please allow camera access.`;
            statusMessage.classList.remove('animate-fade-in-out'); // Stop pulsing on error
            startButton.disabled = false;
            stopButton.disabled = true;
        }
    }

    function stopCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            webcamVideo.srcObject = null;
            clearInterval(detectionInterval);
            detectionInterval = null;
            canvasContext.clearRect(0, 0, emotionCanvas.width, emotionCanvas.height);
            predictionHistory = [];
            currentOverallEmotion = 'No Face Detected';

            startButton.disabled = false;
            stopButton.disabled = true;
            statusMessage.textContent = 'Camera stopped.';
            updateEmotionDisplay('No Face Detected'); // Reset emotion card
        }
    }

    async function sendFrameForDetection() {
        if (!webcamVideo.srcObject || webcamVideo.paused || webcamVideo.ended) {
            return;
        }

        canvasContext.clearRect(0, 0, emotionCanvas.width, emotionCanvas.height);
        canvasContext.drawImage(webcamVideo, 0, 0, videoWidth, videoHeight);

        const imageData = emotionCanvas.toDataURL('image/jpeg', 0.7);

        try {
            const response = await fetch(BACKEND_URL, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ image: imageData }),
            });

            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
            }

            const data = await response.json();
            processAndDisplayPredictions(data.predictions);

        } catch (error) {
            console.error('Error sending frame to backend:', error);
            statusMessage.textContent = 'Error communicating with backend.';
        }
    }

    function processAndDisplayPredictions(rawPredictions) {
        let overallDominantEmotion = 'No Face Detected'; // Default for overall display
        let currentFaceSmoothedEmotion = 'No Face Detected'; // To track the emotion for the first/largest face for canvas drawing

        if (rawPredictions && rawPredictions.length > 0 && rawPredictions[0].bbox) {
            // Focus on the first detected face for overall emotion display and smoothing
            // For multi-face smoothing, you'd need a more complex history tracking per face.
            const firstFacePrediction = rawPredictions[0];
            const currentDominantEmotion = firstFacePrediction.dominant_emotion;
            const currentConfidence = firstFacePrediction.emotion_scores[currentDominantEmotion];

            // Add current prediction to history
            predictionHistory.push({
                dominant_emotion: currentDominantEmotion,
                confidence: currentConfidence
            });

            if (predictionHistory.length > SMOOTHING_WINDOW_SIZE) {
                predictionHistory.shift(); // Remove the oldest prediction
            }

            let smoothedEmotion = 'Undetermined';
            let bestConfidenceSmoothed = 0;

            if (predictionHistory.length > 0) {
                const emotionCounts = {};
                const totalConfidence = {};

                predictionHistory.forEach(hist_p => {
                    emotionCounts[hist_p.dominant_emotion] = (emotionCounts[hist_p.dominant_emotion] || 0) + 1;
                    totalConfidence[hist_p.dominant_emotion] = (totalConfidence[hist_p.dominant_emotion] || 0) + hist_p.confidence;
                });

                let maxCount = 0;
                for (const emotion in emotionCounts) {
                    if (emotionCounts[emotion] > maxCount) {
                        maxCount = emotionCounts[emotion];
                        smoothedEmotion = emotion;
                        bestConfidenceSmoothed = totalConfidence[emotion] / emotionCounts[emotion];
                    } else if (emotionCounts[emotion] === maxCount) {
                        if ((totalConfidence[emotion] / emotionCounts[emotion]) > bestConfidenceSmoothed) {
                            smoothedEmotion = emotion;
                            bestConfidenceSmoothed = totalConfidence[emotion] / emotionCounts[emotion];
                        }
                    }
                }
            }

            // Apply Confidence Threshold
            if (smoothedEmotion !== 'Undetermined' && bestConfidenceSmoothed >= MIN_CONFIDENCE_THRESHOLD) {
                currentFaceSmoothedEmotion = smoothedEmotion;
            } else if (currentConfidence >= MIN_CONFIDENCE_THRESHOLD) {
                currentFaceSmoothedEmotion = currentDominantEmotion; // Fallback to current if confident enough
            } else {
                currentFaceSmoothedEmotion = 'Analyzing...';
            }
            overallDominantEmotion = currentFaceSmoothedEmotion;

            // --- Draw on Canvas for all detected faces ---
            rawPredictions.forEach(p => {
                const [x, y, w, h] = p.bbox;
                const displayEmotionForCanvas = (p.dominant_emotion && p.emotion_scores[p.dominant_emotion] >= MIN_CONFIDENCE_THRESHOLD) ? p.dominant_emotion : 'Analyzing...';
                const color = EMOTION_CANVAS_COLORS[displayEmotionForCanvas] || EMOTION_CANVAS_COLORS['Analyzing...'];

                // Draw bounding box
                canvasContext.strokeStyle = color;
                canvasContext.lineWidth = 2;
                canvasContext.strokeRect(x, y, w, h);

                // Draw emotion label background
                canvasContext.fillStyle = color;
                const fontSize = Math.max(12, Math.min(20, w / 6)); // Dynamic font size based on face width
                canvasContext.font = `${fontSize}px Arial`;
                const textMetrics = canvasContext.measureText(displayEmotionForCanvas);
                const textWidth = textMetrics.width;
                const textHeight = fontSize + 6; // Approx height for background

                canvasContext.fillRect(x, y - textHeight, textWidth + 10, textHeight); // Pad a bit

                // Draw emotion label text
                canvasContext.fillStyle = 'white'; // Text color
                canvasContext.textAlign = 'left';
                canvasContext.fillText(displayEmotionForCanvas, x + 5, y - 5);
            });
        } else {
            overallDominantEmotion = rawPredictions[0]?.message || 'No Face Detected';
            predictionHistory = []; // Clear history if no faces
        }

        updateEmotionDisplay(overallDominantEmotion); // Update the main display
    }

    // --- NEW FUNCTION: Update Emotion Card with Styling and Animation ---
    function updateEmotionDisplay(newEmotion) {
        if (newEmotion === currentOverallEmotion) {
            // If emotion hasn't changed, just ensure it's active and pulsing
            emotionDisplay.classList.add('is-active', 'animate-pulse');
            return;
        }

        // Remove old emotion class and animation class
        emotionDisplay.classList.remove('is-active', 'animate-pulse', `emotion-${currentOverallEmotion.replace(/\s/g, '')}`);

        // Add class for the new emotion
        emotionDisplay.classList.add(`emotion-${newEmotion.replace(/\s/g, '')}`); // Remove spaces for class name

        // Set text content
        emotionDisplayText.textContent = newEmotion;

        // Trigger a subtle fade/scale animation when emotion changes
        // By toggling these classes, we can trigger CSS transitions/animations
        emotionDisplay.classList.remove('opacity-0', 'scale-95'); // Reset for animation
        emotionDisplay.classList.add('is-active'); // Make visible and active

        // Add a small timeout to re-add pulse, ensuring transition plays
        setTimeout(() => {
            if (newEmotion !== 'No Face Detected' && newEmotion !== 'Analyzing...') {
                 emotionDisplay.classList.add('animate-pulse');
            } else {
                 emotionDisplay.classList.remove('animate-pulse');
            }
        }, 50); // Small delay
        currentOverallEmotion = newEmotion;
    }

    // Initial state setup for the emotion card
    updateEmotionDisplay('No Face Detected');
});