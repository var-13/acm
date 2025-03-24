import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import time

def preprocess_image(image):
    """Preprocess the input image for digit recognition."""
    # Convert to grayscale if image is colored
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Increase contrast
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    
    # Apply adaptive thresholding with different parameters
    image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 15, 5)
    
    # Remove noise
    kernel = np.ones((2,2), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    
    # Resize to 28x28
    image = cv2.resize(image, (28, 28))
    
    # Normalize
    image = image.astype('float32') / 255.0
    
    # Add padding to center the digit
    image = np.pad(image, 4, mode='constant')
    image = cv2.resize(image, (28, 28))
    
    # Flatten the image
    image = image.flatten()
    
    return image

def train_models():
    """Train SVM and KNN models on a subset of MNIST dataset."""
    print("Loading MNIST dataset (this may take a few seconds)...")
    
    # Load MNIST dataset
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    # Use only 10,000 samples for faster training
    sample_size = 10000
    indices = np.random.choice(len(X), sample_size, replace=False)
    X = X[indices]
    y = y[indices]
    
    print(f"Using {sample_size} samples for training...")
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining SVM model (this will take about 2-3 minutes)...")
    svm = SVC(kernel='rbf', cache_size=1000)  # Increased cache size for faster training
    svm.fit(X_train, y_train)
    
    print("\nTraining KNN model (this will take about 30 seconds)...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    svm_pred = svm.predict(X_test)
    knn_pred = knn.predict(X_test)
    
    print("\nModel Performance:")
    print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
    print("KNN Accuracy:", accuracy_score(y_test, knn_pred))
    
    return svm, knn

def real_time_recognition(svm, knn):
    """Perform real-time digit recognition using webcam."""
    cap = cv2.VideoCapture(0)
    
    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\nStarting real-time recognition...")
    print("Instructions:")
    print("1. Write a digit in the green rectangle")
    print("2. Use a dark pen or marker on white paper")
    print("3. Hold your hand steady for better recognition")
    print("4. Press 'q' to quit")
    print("5. Press 'r' to reset predictions")
    
    # Variables for prediction stability
    last_prediction_time = 0
    prediction_delay = 0.5  # seconds
    last_svm_pred = None
    last_knn_pred = None
    
    # Create a combined display window
    cv2.namedWindow('Digit Recognition System', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Digit Recognition System', 1200, 400)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Create a copy of the frame for drawing
        display = frame.copy()
        
        # Define region of interest (ROI) - made larger
        roi = frame[50:350, 50:350]  # Increased ROI size
        
        # Draw rectangle around ROI
        cv2.rectangle(display, (50, 50), (350, 350), (0, 255, 0), 2)
        
        # Add instructions on screen
        cv2.putText(display, "Write digit here", (50, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Get current time
        current_time = time.time()
        
        # Only make predictions after delay
        if current_time - last_prediction_time >= prediction_delay:
            # Preprocess ROI
            processed = preprocess_image(roi)
            
            # Make predictions
            svm_pred = svm.predict([processed])[0]
            knn_pred = knn.predict([processed])[0]
            
            # Update last predictions
            last_svm_pred = svm_pred
            last_knn_pred = knn_pred
            last_prediction_time = current_time
            
            # Create processed image display (convert to 3 channels)
            processed_display = processed.reshape(28, 28)
            processed_display = (processed_display * 255).astype(np.uint8)
            processed_display = cv2.resize(processed_display, (200, 200))
            processed_display = cv2.cvtColor(processed_display, cv2.COLOR_GRAY2BGR)
            
            # Create a black background for predictions (3 channels)
            pred_display = np.zeros((200, 200, 3), dtype=np.uint8)
            cv2.putText(pred_display, f"SVM: {last_svm_pred}", (20, 80), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(pred_display, f"KNN: {last_knn_pred}", (20, 160), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Resize webcam feed to match height of other displays
            display = cv2.resize(display, (400, 200))
            
            # Combine all displays
            combined_display = np.hstack((display, processed_display, pred_display))
            cv2.imshow('Digit Recognition System', combined_display)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            last_svm_pred = None
            last_knn_pred = None
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    print("Starting Handwritten Digit Recognition System...")
    print("This version uses a smaller dataset for faster training.")
    
    # Train models
    svm, knn = train_models()
    
    # Start real-time recognition
    real_time_recognition(svm, knn)

if __name__ == "__main__":
    main() 