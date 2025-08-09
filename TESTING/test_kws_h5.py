import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
import threading
import time
import os

# Constants
SAMPLE_RATE = 16000
DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
MFCC_N = 40
MAX_FRAMES = 49
MODEL_PATH = "E:\kwsspotter_V2\TESTING\kws_model.h5"

class KeywordSpotter:
    def __init__(self, model_path):
        """Initialize the keyword spotter with a trained model"""
        self.model = keras.models.load_model(model_path)
        self.is_listening = False
        self.detection_threshold = 0.5  # Adjust this threshold as needed
        
    def extract_mfcc(self, audio_data, sr=16000, n_mfcc=40):
        """Extract MFCC features from audio data"""
        # Ensure audio is the right length
        if len(audio_data) > sr:
            audio_data = audio_data[:sr]
        else:
            audio_data = np.pad(audio_data, (0, sr - len(audio_data)))
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc)
        
        # Ensure consistent shape (44, 40) for CNN input
        if mfcc.shape[1] < 44:
            mfcc = np.pad(mfcc, ((0, 0), (0, 44 - mfcc.shape[1])))
        elif mfcc.shape[1] > 44:
            mfcc = mfcc[:, :44]
        
        return mfcc.T  # Shape: (44, 40)
    
    def predict_keyword(self, audio_data):
        """Predict if the audio contains the keyword"""
        try:
            # Extract MFCC features
            mfcc = self.extract_mfcc(audio_data)
            
            # Reshape for model input (batch_size, height, width, channels)
            mfcc = mfcc.reshape(1, 44, 40, 1)
            
            # Make prediction
            prediction = self.model.predict(mfcc, verbose=0)
            confidence = prediction[0][0]
            
            return confidence > self.detection_threshold, confidence
            
        except Exception as e:
            print(f"❌ Error during prediction: {e}")
            return False, 0.0
    
    def record_audio_chunk(self):
        """Record a 1-second audio chunk"""
        try:
            audio = sd.rec(NUM_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            return audio.flatten()
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
            return None
    
    def start_listening(self):
        """Start continuous listening for keywords"""
        if not os.path.exists(MODEL_PATH):
            print(f"❌ Model file not found: {MODEL_PATH}")
            print("Please train the model first using costomkws_v2.py")
            return
        
        print("🎧 Starting keyword detection...")
        print("📋 Instructions:")
        print("   - Speak your keyword clearly")
        print("   - The system will continuously listen")
        print("   - Press Ctrl+C to stop")
        print(f"   - Detection threshold: {self.detection_threshold}")
        print()
        
        self.is_listening = True
        detection_count = 0
        
        try:
            while self.is_listening:
                # Record audio chunk
                audio_data = self.record_audio_chunk()
                
                if audio_data is not None:
                    # Predict keyword
                    is_keyword, confidence = self.predict_keyword(audio_data)
                    
                    if is_keyword:
                        detection_count += 1
                        print(f"🎯 KEYWORD DETECTED! (Confidence: {confidence:.3f}) - Detection #{detection_count}")
                        # Optional: Add a small delay to avoid multiple detections
                        time.sleep(0.5)
                    else:
                        # Print confidence occasionally for debugging
                        if confidence > 0.3:  # Only print if confidence is somewhat high
                            print(f"🔍 Listening... (Confidence: {confidence:.3f})")
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping keyword detection...")
        except Exception as e:
            print(f"❌ Error during listening: {e}")
        finally:
            self.is_listening = False
            print(f"📊 Total detections: {detection_count}")

def main():
    print("🚀 Keyword Spotting Test (H5 Model)")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please train the model first using costomkws_v2.py")
        return
    
    # Initialize keyword spotter
    spotter = KeywordSpotter(MODEL_PATH)
    
    # Ask user for detection threshold
    try:
        threshold = float(input(f"Enter detection threshold (0.0-1.0, default {spotter.detection_threshold}): ") or spotter.detection_threshold)
        spotter.detection_threshold = max(0.0, min(1.0, threshold))
    except ValueError:
        print(f"Using default threshold: {spotter.detection_threshold}")
    
    # Start listening
    spotter.start_listening()

if __name__ == "__main__":
    main()
