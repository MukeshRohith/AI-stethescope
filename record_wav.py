import serial
import wave
import struct
import time

# --- Settings ---
SERIAL_PORT = 'COM7'  # <--- CHANGE TO YOUR ESP32 PORT
BAUD_RATE = 921600    # Matches the new ESP32 code
SAMPLE_RATE = 16000
RECORD_SECONDS = 5
OUTPUT_FILENAME = "my_live_heartbeat.wav"

def record_audio():
    print(f"Connecting to ESP32 on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    except Exception as e:
        print(f"Error opening port: {e}")
        return

    print("Place stethoscope on chest. Recording starts in 3 seconds...")
    time.sleep(3)
    
    print(f"🔴 RECORDING FOR {RECORD_SECONDS} SECONDS... STAY STILL!")
    
    ser.flushInput()
    audio_data = []
    total_samples = SAMPLE_RATE * RECORD_SECONDS
    
    # Collect data from the serial port
    while len(audio_data) < total_samples:
        try:
            line = ser.readline().decode('utf-8').strip()
            if line:
                # Convert string to integer
                sample = int(float(line))
                
                # Clip the integer to standard 16-bit audio limits
                sample = max(-32768, min(32767, sample))
                audio_data.append(sample)
                
        except ValueError:
            pass # Ignore corrupted serial lines
        except Exception as e:
            print(f"Serial read error: {e}")
            break

    print("✅ Recording complete! Saving to WAV...")
    ser.close()

    # --- Save as .wav file ---
    with wave.open(OUTPUT_FILENAME, 'w') as wav_file:
        # 1 channel (mono), 2 bytes per sample (16-bit), 16000 Hz
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        
        # Pack the integers into binary data for the wav file
        for sample in audio_data:
            data = struct.pack('<h', sample) # '<h' means little-endian 16-bit short
            wav_file.writeframesraw(data)

    print(f"File saved successfully as: {OUTPUT_FILENAME}")
    print("You can now open it with VLC, Windows Media Player, or Audacity!")

if __name__ == "__main__":
    record_audio()