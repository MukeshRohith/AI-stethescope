import os
import glob
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

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

# --- 3. Plotting Professional Waveforms ---
def plot_waveforms(healthy_dir, unhealthy_dir):
    """Plots one healthy and one unhealthy waveform for the final report."""
    
    # Force Matplotlib out of dark mode
    plt.style.use('default') 
    
    # Grab the first file from each directory
    healthy_file = glob.glob(os.path.join(healthy_dir, "*.wav"))[0]
    unhealthy_file = glob.glob(os.path.join(unhealthy_dir, "*.wav"))[0]
    
    # Load and filter
    h_audio, sr = librosa.load(healthy_file, sr=16000)
    uh_audio, _ = librosa.load(unhealthy_file, sr=16000)
    
    h_filtered = bandpass_filter(h_audio, sr)
    uh_filtered = bandpass_filter(uh_audio, sr)
    
    # Create a time axis for plotting (limiting to 3 seconds for readability)
    duration = min(len(h_filtered), len(uh_filtered), 3 * sr)
    t = np.linspace(0, duration / sr, duration)
    
    # Plotting: Explicitly setting facecolor to 'white'
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, facecolor='white')
    fig.suptitle('Filtered Phonocardiogram (PCG) Comparison', fontsize=14, fontweight='bold', color='black')
    
    # Normal Plot
    ax1.set_facecolor('white')
    ax1.plot(t, h_filtered[:duration], color='#0033a0', linewidth=1.2) # Deep blue for healthy
    ax1.set_title('Healthy Heart Sound (Normal S1/S2)', fontsize=11, color='black')
    ax1.set_ylabel('Amplitude', color='black')
    ax1.grid(True, linestyle='--', color='#cccccc', alpha=0.8) # Light grey grid
    ax1.tick_params(colors='black')
    
    # Abnormal Plot
    ax2.set_facecolor('white')
    ax2.plot(t, uh_filtered[:duration], color='#b30000', linewidth=1.2) # Deep red for unhealthy
    ax2.set_title('Unhealthy Heart Sound (Murmur / Anomaly)', fontsize=11, color='black')
    ax2.set_xlabel('Time (Seconds)', color='black')
    ax2.set_ylabel('Amplitude', color='black')
    ax2.grid(True, linestyle='--', color='#cccccc', alpha=0.8) # Light grey grid
    ax2.tick_params(colors='black')
    
    # Adjust layout to prevent clipping
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Generating Professional Plots...")
    plot_waveforms(HEALTHY_DIR, UNHEALTHY_DIR)