import os
import glob
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib 

# --- 1. Define File Paths ---
HEALTHY_DIR = r"C:\Users\Save Trees\Downloads\archive(8)\heart_sound\train\healthy"
UNHEALTHY_DIR = r"C:\Users\Save Trees\Downloads\archive(8)\heart_sound\train\unhealthy"

# --- 2. Digital Signal Processing (Band-Pass Filter) ---
def bandpass_filter(data, fs, lowcut=20.0, highcut=200.0, order=4):
    """Filters the audio to the 20-200Hz heart sound range."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --- 3. Feature Extraction for Machine Learning ---
def extract_features(file_path):
    """Loads audio, applies filter, and extracts MFCC features."""
    try:
        # Load audio at a fixed sample rate (16kHz matches your ESP32)
        audio, sr = librosa.load(file_path, sr=16000)
        
        # Apply the same bandpass filter your hardware uses
        filtered_audio = bandpass_filter(audio, sr)
        
        # Extract Mel-Frequency Cepstral Coefficients (MFCCs)
        mfccs = librosa.feature.mfcc(y=filtered_audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- 4. Main Execution: Train the Model ---
def main():
    print("Extracting features from audio files...")
    X = [] # Features
    y = [] # Labels (0 = Healthy, 1 = Unhealthy)
    
    # Process Healthy Files
    for file in glob.glob(os.path.join(HEALTHY_DIR, "*.wav")):
        features = extract_features(file)
        if features is not None:
            X.append(features)
            y.append(0)
            
    # Process Unhealthy Files
    for file in glob.glob(os.path.join(UNHEALTHY_DIR, "*.wav")):
        features = extract_features(file)
        if features is not None:
            X.append(features)
            y.append(1)
            
    X = np.array(X)
    y = np.array(y)
    
    print(f"Total samples processed: {len(X)}")
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and Train the Random Forest Classifier
    print("\nTraining Random Forest Model...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Evaluate the Model
    print("Evaluating Model on Test Data...\n")
    y_pred = clf.predict(X_test)
    
    print("=== Model Performance Report ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    print(classification_report(y_test, y_pred, target_names=['Healthy (0)', 'Unhealthy (1)']))
    
    # Save the Trained Model
    print("Saving the trained model...")
    joblib.dump(clf, 'pcg_classifier_model.pkl')
    print("Model saved successfully as pcg_classifier_model.pkl!")

if __name__ == "__main__":
    main()