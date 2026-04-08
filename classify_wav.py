import numpy as np
import librosa
import joblib
from scipy.signal import butter, filtfilt

# --- 1. Define File Paths ---
#AUDIO_FILE = r"D:\heartbeat classifier\my_live_heartbeat.wav"
AUDIO_FILE = r"C:\Users\Save Trees\Downloads\steth_recording (2).wav"

# Point this to the trained model you saved earlier
MODEL_FILE = 'pcg_classifier_model.pkl' 

# --- 2. Digital Signal Processing (Band-Pass Filter) ---
# This MUST be identical to the filter used during training
def bandpass_filter(data, fs, lowcut=20.0, highcut=200.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# --- 3. Feature Extraction ---
# This MUST be identical to the extraction used during training
def extract_features(file_path):
    print(f"[*] Loading audio file: {file_path}")
    try:
        # Load audio at exactly 16kHz
        audio, sr = librosa.load(file_path, sr=16000)
        
        # Apply the bandpass filter
        filtered_audio = bandpass_filter(audio, sr)
        
        # Extract the 13 MFCCs
        mfccs = librosa.feature.mfcc(y=filtered_audio, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        
        return mfccs_mean
    except Exception as e:
        print(f"[!] Error processing audio: {e}")
        return None

# --- 4. Main Classification Routine ---
def main():
    print("========================================")
    print("   AI Stethoscope Diagnosis Module      ")
    print("========================================\n")
    
    # Step A: Load the AI Model
    print("[*] Loading trained Random Forest model...")
    try:
        clf = joblib.load(MODEL_FILE)
    except FileNotFoundError:
        print(f"[!] ERROR: Could not find '{MODEL_FILE}'. Make sure it is in the same folder!")
        return

    # Step B: Process the recorded heartbeat
    print("[*] Analyzing acoustic signatures...")
    features = extract_features(AUDIO_FILE)
    
    if features is None:
        print("[!] Diagnosis aborted due to audio read error.")
        return
        
    # Scikit-Learn expects a 2D array for predictions (e.g., multiple samples).
    # Since we only have one heartbeat sample, we reshape it to (1, -1)
    features_reshaped = features.reshape(1, -1)
    
    # Step C: Ask the AI for a diagnosis
    print("[*] Running classification...\n")
    prediction = clf.predict(features_reshaped)[0]
    
    # Step D: Print the result!
    print("========================================")
    if prediction == 0:
        print("   DIAGNOSIS: [ HEALTHY ] (Normal S1/S2)")
        print("   The algorithm detected no anomalies.")
    else:
        print("   DIAGNOSIS: [ ABNORMAL ] (Murmur/Anomaly)")
        print("   The algorithm detected irregular patterns.")
    print("========================================")

if __name__ == "__main__":
    main()