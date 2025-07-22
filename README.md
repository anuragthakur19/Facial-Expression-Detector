# Real-time Human Expression Detector

## Project Overview

This project implements a real-time human facial expression detection system using a deep learning model. It captures live video input from a camera, processes frames to detect faces, infers emotions, and displays the results with dynamic visual feedback. The system is designed to be highly responsive and user-friendly, providing an interactive experience.

## Key Features

* **Real-time Emotion Detection:** Identifies 7 universal emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) in live video streams.
* **Deep Learning Model:** Utilizes a Convolutional Neural Network (CNN) trained on the FER-2013 dataset.
* **Web-based Frontend:** Interactive and responsive user interface built with HTML, CSS (Tailwind CSS), and JavaScript.
* **Python Backend (Flask):** Serves the trained AI model, handles image processing, and performs emotion inference.
* **Temporal Smoothing:** Incorporates logic to stabilize emotion predictions over time, reducing flicker and improving perceived accuracy.
* **Confidence Thresholding:** Filters out low-confidence predictions for clearer results.
* **Dynamic UI:** Features animations and styling that adapt based on detected emotions.

## Technical Stack

* **Frontend:** HTML5, CSS3 (Tailwind CSS), JavaScript, WebRTC
* **Backend:** Python 3.x, Flask, TensorFlow/Keras, OpenCV
* **Model Training:** Python, TensorFlow/Keras, scikit-learn, Pandas (on Google Colab)

## Important Note on Model Calibration

**It is important to note that the underlying deep learning model exhibits a heightened sensitivity to subtle facial changes. This characteristic stems from its training on a comprehensive dataset, which has resulted in a finely tuned ability to detect even minor variations in expression.**

## Setup and Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/your-repo-name.git](https://github.com/YourUsername/your-repo-name.git)
    cd your-repo-name
    ```
    *(Remember to replace `YourUsername/your-repo-name` with your actual GitHub repository details)*

2.  **Set up a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # For Windows PowerShell:
    .\venv\Scripts\Activate.ps1
    # For macOS/Linux or Git Bash:
    source venv/bin/activate
    ```

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Obtain the Trained Model:**
    Ensure you have the `best_emotion_model.h5` file in your `backend/` directory. This file is typically generated during the model training phase (e.g., from the provided Google Colab notebook). If it's not present, you might need to run the training notebook to generate it.

## Usage

1.  **Start the Backend Server:**
    Open a terminal, activate your virtual environment, navigate to the `backend/` directory, and run the Flask application:
    ```bash
    # In your project root:
    cd backend
    python app.py
    ```
    Keep this terminal running.

2.  **Open the Frontend in your Browser:**
    Open another terminal, activate your virtual environment, navigate to the `frontend/` directory, and open the `index.html` file in your web browser.
    ```bash
    # In your project root:
    cd frontend
    start index.html # On Windows
    # or if using a Live Server extension in VS Code, open the folder and use "Go Live"
    ```

3.  **Grant Camera Access & Detect Emotions:**
    * Allow your browser to access your webcam when prompted.
    * Click the "Start Camera" button on the webpage.
    * Display various facial expressions (Happy, Sad, Angry, etc.) in front of the camera and observe the real-time detection and dynamic UI updates.

## Contributing

Feel free to fork the repository, open issues, or submit pull requests.

---
