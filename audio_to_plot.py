import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


AUDIO_FILE_PATH = r"D:\heartbeat classifier\my_live_heartbeat.wav"

def bandpass_filter(data, fs, lowcut=20.0, highcut=200.0, order=4):
    """Filters the audio to the 20-200Hz heart sound range."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def plot_single_waveform(file_path):
    print(f"Loading audio from: {file_path}")
    
    try:
        # Load audio at 16kHz to match ESP32
        audio, sr = librosa.load(file_path, sr=16000)
    except Exception as e:
        print(f"Error loading file. Please double-check the path!\nDetails: {e}")
        return
        
    # Apply the bandpass filter
    filtered_audio = bandpass_filter(audio, sr)
    
    # Limit to the first 5 seconds for visual readability
    duration = min(len(filtered_audio), 5 * sr)
    t = np.linspace(0, duration / sr, duration)
    
    # Force Matplotlib to a clean, white background
    plt.style.use('default') 
    
    # Set up the graph
    fig, ax = plt.subplots(figsize=(10, 4), facecolor='white')
    ax.set_facecolor('white')
    
    # Plot the wave in a professional deep blue
    ax.plot(t, filtered_audio[:duration], color='#0033a0', linewidth=1.2) 
    
    # Extract the file name dynamically for the graph title
    file_name = os.path.basename(file_path)
    ax.set_title(f'Filtered Phonocardiogram (PCG) - {file_name}', fontsize=12, fontweight='bold', color='black')
    ax.set_xlabel('Time (Seconds)', color='black')
    ax.set_ylabel('Amplitude', color='black')
    ax.grid(True, linestyle='--', color='#cccccc', alpha=0.8)
    ax.tick_params(colors='black')
    
    # Display the plot
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_single_waveform(AUDIO_FILE_PATH)