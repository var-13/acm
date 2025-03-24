# Handwritten Digit Recognition

This project implements a Handwritten Digit Recognition system using OpenCV and Machine Learning techniques (SVM and KNN classifiers). The system can recognize digits from 0 to 9 from handwritten images.

## Features
- Preprocessing of handwritten digit images using OpenCV
- Implementation of both SVM and KNN classifiers
- Training on MNIST dataset
- Real-time digit recognition from webcam
- Performance comparison between SVM and KNN

## Requirements
- Python 3.7+
- OpenCV
- NumPy
- scikit-learn
- Matplotlib

## Installation
1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
1. Run the main script:
```bash
python digit_recognition.py
```

2. The program will:
   - Load and preprocess the MNIST dataset
   - Train both SVM and KNN models
   - Show performance metrics
   - Open webcam for real-time digit recognition

## Project Structure
- `digit_recognition.py`: Main implementation file
- `requirements.txt`: Project dependencies
- `data/`: Directory for storing MNIST dataset
- `README.md`: Project documentation

## How it Works
1. Image Preprocessing:
   - Resize images to 28x28 pixels
   - Apply thresholding
   - Normalize pixel values

2. Model Training:
   - SVM: Support Vector Machine classifier
   - KNN: K-Nearest Neighbors classifier

3. Real-time Recognition:
   - Capture webcam feed
   - Preprocess frames
   - Predict digits using trained models 

[Webcam Feed] [Processed Image] [Predictions] 