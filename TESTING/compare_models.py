import numpy as np
import librosa
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
import time
import os

# Constants
SAMPLE_RATE = 16000
DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
MFCC_N = 40
H5_MODEL_PATH = "kws_model.h5"
TFLITE_MODEL_PATH = "kws_model.tflite"

class ModelComparison:
    def __init__(self):
        """Initialize both H5 and TFLite models for comparison"""
        self.h5_model = None
        self.tflite_interpreter = None
        self.detection_threshold = 0.5
        
        # Load H5 model
        if os.path.exists(H5_MODEL_PATH):
            self.h5_model = keras.models.load_model(H5_MODEL_PATH)
            print(f"✅ H5 model loaded: {H5_MODEL_PATH}")
        else:
            print(f"❌ H5 model not found: {H5_MODEL_PATH}")
        
        # Load TFLite model
        if os.path.exists(TFLITE_MODEL_PATH):
            self.tflite_interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
            self.tflite_interpreter.allocate_tensors()
            self.input_details = self.tflite_interpreter.get_input_details()
            self.output_details = self.tflite_interpreter.get_output_details()
            print(f"✅ TFLite model loaded: {TFLITE_MODEL_PATH}")
        else:
            print(f"❌ TFLite model not found: {TFLITE_MODEL_PATH}")
    
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
    
    def predict_h5(self, audio_data):
        """Predict using H5 model"""
        if self.h5_model is None:
            return False, 0.0, 0.0
        
        try:
            mfcc = self.extract_mfcc(audio_data)
            mfcc = mfcc.reshape(1, 44, 40, 1)
            
            start_time = time.time()
            prediction = self.h5_model.predict(mfcc, verbose=0)
            inference_time = time.time() - start_time
            
            confidence = prediction[0][0]
            return confidence > self.detection_threshold, confidence, inference_time
            
        except Exception as e:
            print(f"❌ H5 prediction error: {e}")
            return False, 0.0, 0.0
    
    def predict_tflite(self, audio_data):
        """Predict using TFLite model"""
        if self.tflite_interpreter is None:
            return False, 0.0, 0.0
        
        try:
            mfcc = self.extract_mfcc(audio_data)
            mfcc = mfcc.reshape(1, 44, 40, 1)
            
            # Ensure correct data type
            input_dtype = self.input_details[0]['dtype']
            mfcc = mfcc.astype(input_dtype)
            
            # Set input tensor
            self.tflite_interpreter.set_tensor(self.input_details[0]['index'], mfcc)
            
            # Run inference
            start_time = time.time()
            self.tflite_interpreter.invoke()
            inference_time = time.time() - start_time
            
            # Get output
            prediction = self.tflite_interpreter.get_tensor(self.output_details[0]['index'])
            confidence = prediction[0][0]
            
            return confidence > self.detection_threshold, confidence, inference_time
            
        except Exception as e:
            print(f"❌ TFLite prediction error: {e}")
            return False, 0.0, 0.0
    
    def record_audio_chunk(self):
        """Record a 1-second audio chunk"""
        try:
            audio = sd.rec(NUM_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
            sd.wait()
            return audio.flatten()
        except Exception as e:
            print(f"❌ Error recording audio: {e}")
            return None
    
    def compare_models(self):
        """Compare both models in real-time"""
        if self.h5_model is None and self.tflite_interpreter is None:
            print("❌ No models available for comparison")
            return
        
        print("🎧 Starting model comparison...")
        print("📋 Instructions:")
        print("   - Speak your keyword clearly")
        print("   - Both models will process the same audio")
        print("   - Press Ctrl+C to stop")
        print(f"   - Detection threshold: {self.detection_threshold}")
        print()
        
        detection_count = 0
        h5_times = []
        tflite_times = []
        
        try:
            while True:
                # Record audio chunk
                audio_data = self.record_audio_chunk()
                
                if audio_data is not None:
                    detection_count += 1
                    print(f"\n🎵 Processing chunk #{detection_count}")
                    
                    # Test H5 model
                    if self.h5_model is not None:
                        h5_detected, h5_conf, h5_time = self.predict_h5(audio_data)
                        h5_times.append(h5_time)
                        h5_status = "🎯 DETECTED" if h5_detected else "❌ Not detected"
                        print(f"   H5 Model: {h5_status} (Confidence: {h5_conf:.3f}, Time: {h5_time:.3f}s)")
                    
                    # Test TFLite model
                    if self.tflite_interpreter is not None:
                        tflite_detected, tflite_conf, tflite_time = self.predict_tflite(audio_data)
                        tflite_times.append(tflite_time)
                        tflite_status = "🎯 DETECTED" if tflite_detected else "❌ Not detected"
                        print(f"   TFLite:  {tflite_status} (Confidence: {tflite_conf:.3f}, Time: {tflite_time:.3f}s)")
                    
                    # Compare results
                    if self.h5_model is not None and self.tflite_interpreter is not None:
                        if h5_detected != tflite_detected:
                            print("   ⚠️  Models disagree!")
                        elif h5_detected and tflite_detected:
                            print("   ✅ Both models detected keyword")
                    
                    # Add delay to avoid overwhelming output
                    time.sleep(0.5)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopping comparison...")
        except Exception as e:
            print(f"❌ Error during comparison: {e}")
        finally:
            # Print summary statistics
            print(f"\n📊 Comparison Summary:")
            print(f"   Total chunks processed: {detection_count}")
            
            if h5_times:
                avg_h5_time = np.mean(h5_times)
                print(f"   H5 average inference time: {avg_h5_time:.3f}s")
            
            if tflite_times:
                avg_tflite_time = np.mean(tflite_times)
                print(f"   TFLite average inference time: {avg_tflite_time:.3f}s")
            
            if h5_times and tflite_times:
                speedup = avg_h5_time / avg_tflite_time if avg_tflite_time > 0 else 0
                print(f"   TFLite speedup: {speedup:.2f}x")

def main():
    print("🚀 Model Comparison Tool")
    print("=" * 30)
    
    # Check if models exist
    h5_exists = os.path.exists(H5_MODEL_PATH)
    tflite_exists = os.path.exists(TFLITE_MODEL_PATH)
    
    if not h5_exists and not tflite_exists:
        print("❌ No models found!")
        print("Please train the model first using costomkws_v2.py")
        return
    
    print(f"📁 H5 model: {'✅' if h5_exists else '❌'}")
    print(f"📁 TFLite model: {'✅' if tflite_exists else '❌'}")
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Ask user for detection threshold
    try:
        threshold = float(input(f"\nEnter detection threshold (0.0-1.0, default {comparison.detection_threshold}): ") or comparison.detection_threshold)
        comparison.detection_threshold = max(0.0, min(1.0, threshold))
    except ValueError:
        print(f"Using default threshold: {comparison.detection_threshold}")
    
    # Start comparison
    comparison.compare_models()

if __name__ == "__main__":
    main()
