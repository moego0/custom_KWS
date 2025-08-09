import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
import threading
import time
import os

# Constants
SAMPLE_RATE = 16000
DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
MFCC_N = 40
MAX_FRAMES = 49
MODEL_PATH = "kws_model.tflite"

class KeywordSpotterTFLite:
    def __init__(self, model_path):
        """Initialize the keyword spotter with a trained TFLite model"""
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        
        # Get input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        self.is_listening = False
        self.detection_threshold = 0.5  # Adjust this threshold as needed
        
        print(f"📱 TFLite model loaded successfully")
        print(f"   Input shape: {self.input_details[0]['shape']}")
        print(f"   Output shape: {self.output_details[0]['shape']}")
        
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
        """Predict if the audio contains the keyword using TFLite model"""
        try:
            # Extract MFCC features
            mfcc = self.extract_mfcc(audio_data)
            
            # Reshape for model input (batch_size, height, width, channels)
            mfcc = mfcc.reshape(1, 44, 40, 1)
            
            # Ensure correct data type
            input_dtype = self.input_details[0]['dtype']
            mfcc = mfcc.astype(input_dtype)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], mfcc)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            prediction = self.interpreter.get_tensor(self.output_details[0]['index'])
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
        
        print("🎧 Starting keyword detection (TFLite)...")
        print("📋 Instructions:")
        print("   - Speak your keyword clearly")
        print("   - The system will continuously listen")
        print("   - Press Ctrl+C to stop")
        print(f"   - Detection threshold: {self.detection_threshold}")
        print()
        
        self.is_listening = True
        detection_count = 0
        inference_times = []
        
        try:
            while self.is_listening:
                # Record audio chunk
                audio_data = self.record_audio_chunk()
                
                if audio_data is not None:
                    # Time the inference
                    start_time = time.time()
                    
                    # Predict keyword
                    is_keyword, confidence = self.predict_keyword(audio_data)
                    
                    inference_time = time.time() - start_time
                    inference_times.append(inference_time)
                    
                    if is_keyword:
                        detection_count += 1
                        print(f"🎯 KEYWORD DETECTED! (Confidence: {confidence:.3f}, Time: {inference_time:.3f}s) - Detection #{detection_count}")
                        # Optional: Add a small delay to avoid multiple detections
                        time.sleep(0.5)
                    else:
                        # Print confidence occasionally for debugging
                        if confidence > 0.3:  # Only print if confidence is somewhat high
                            print(f"🔍 Listening... (Confidence: {confidence:.3f}, Time: {inference_time:.3f}s)")
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping keyword detection...")
        except Exception as e:
            print(f"❌ Error during listening: {e}")
        finally:
            self.is_listening = False
            if inference_times:
                avg_time = np.mean(inference_times)
                print(f"📊 Total detections: {detection_count}")
                print(f"📊 Average inference time: {avg_time:.3f}s")

def main():
    print("🚀 Keyword Spotting Test (TFLite Model)")
    print("=" * 45)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please train the model first using costomkws_v2.py")
        return
    
    # Initialize keyword spotter
    spotter = KeywordSpotterTFLite(MODEL_PATH)
    
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
