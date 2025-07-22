# -*- coding: utf-8 -*-
"""
# Real-time Human Expression Detection Backend

This Flask application serves as the backend for real-time human expression detection.
It loads a pre-trained Keras model, receives image frames from a frontend,
performs face detection and emotion prediction, and returns the results.

**Pre-requisites:**
1.  Python 3.x installed.
2.  `best_emotion_model.h5` (your trained model) in the same directory as this script.
3.  Install necessary libraries:
    `pip install Flask tensorflow opencv-python numpy Pillow flask-cors`
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
import base64
from PIL import Image
import io

app = Flask(__name__)
CORS(app) # Enable CORS for all routes, allowing frontend to access from different origin

# --- 1. Load the Trained Model and Face Detector ---
print("Loading model and face detector...")
try:
    # Load the Keras model
    model = tf.keras.models.load_model('best_emotion_model.h5')
    print("Model loaded successfully: best_emotion_model.h5")

    # Load OpenCV's Haar Cascade for face detection
    # Ensure you have 'haarcascade_frontalface_default.xml' in your OpenCV data path
    # or specify the full path if it's stored elsewhere.
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_classifier.empty():
        raise IOError('Unable to load the face cascade classifier xml file.')
    print("Face cascade classifier loaded successfully.")

    # Define emotion labels (consistent with your training)
    emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    image_size = (48, 48) # Model input size

except Exception as e:
    print(f"Error loading resources: {e}")
    # Exit or handle gracefully if resources can't be loaded
    exit()

# --- 2. Define the Prediction Endpoint ---
@app.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    if 'image' not in request.json:
        return jsonify({"error": "No image data provided"}), 400

    image_data_b64 = request.json['image']
    # Remove the "data:image/jpeg;base64," prefix if it exists
    if "base64," in image_data_b64:
        image_data_b64 = image_data_b64.split("base64,")[1]

    try:
        # Decode base64 image data
        image_bytes = base64.b64decode(image_data_b64)
        # Convert bytes to a PIL Image
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        # Convert PIL Image to OpenCV format (BGR)
        frame = np.array(pil_image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"Error decoding image: {e}")
        return jsonify({"error": "Invalid image data"}), 400

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

    predictions = []
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi_gray = gray_frame[y:y+h, x:x+w]
            # Resize to model's expected input size (48x48)
            face_roi_resized = cv2.resize(face_roi_gray, image_size, interpolation=cv2.INTER_AREA)
            # Normalize and expand dimensions for model input
            face_roi_normalized = face_roi_resized / 255.0
            face_roi_input = np.expand_dims(np.expand_dims(face_roi_normalized, -1), 0) # (1, 48, 48, 1)

            # Make prediction
            emotion_prediction = model.predict(face_roi_input)[0]
            dominant_emotion_index = np.argmax(emotion_prediction)
            dominant_emotion = emotion_labels[dominant_emotion_index]

            # Get all emotion scores
            emotion_scores = {emotion_labels[i]: float(emotion_prediction[i]) for i in range(len(emotion_labels))}

            predictions.append({
                "bbox": [int(x), int(y), int(w), int(h)], # Bounding box coordinates
                "dominant_emotion": dominant_emotion,
                "emotion_scores": emotion_scores
            })
    else:
        # If no faces detected, you might want to send a specific message
        predictions.append({
            "message": "No face detected"
        })


    return jsonify({"success": True, "predictions": predictions})

# --- 3. Run the Flask App ---
if __name__ == '__main__':
    # Run the app. Debug mode is useful for development, disable in production.
    # Host '0.0.0.0' makes the server accessible from other devices on the network.
    app.run(host='0.0.0.0', port=5000, debug=True)
    print("Flask app started on http://0.0.0.0:5000")