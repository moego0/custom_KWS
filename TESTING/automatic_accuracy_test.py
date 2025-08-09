import os
import numpy as np
import librosa
import soundfile as sf
import tensorflow as tf
from tensorflow import keras
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Constants
SAMPLE_RATE = 16000
DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
MODEL_PATH = "E:/kwsspotter_V2/TESTING/kws_model.h5"
TEST_DATA_DIR = "E:/kwsspotter_V2/TESTING/test_recordings"
BACKGROUND_NOISE_DIR = "E:/kwsspotter_V2/negative_samples"

class AutomaticAccuracyTester:
    def __init__(self, model_path):
        """Initialize the accuracy tester with a trained model"""
        self.model = keras.models.load_model(model_path)
        self.detection_threshold = 0.5
        self.results = {
            'true_positives': 0,
            'false_positives': 0,
            'true_negatives': 0,
            'false_negatives': 0,
            'predictions': [],
            'true_labels': [],
            'confidences': []
        }
        
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
    
    def predict_sample(self, audio_data):
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
    
    def test_positive_samples(self):
        """Test the model on positive samples (keyword 'yes')"""
        print("🔍 Testing positive samples (keyword 'yes')...")
        
        if not os.path.exists(TEST_DATA_DIR):
            print(f"❌ Test data directory not found: {TEST_DATA_DIR}")
            return 0
        
        wav_files = [f for f in os.listdir(TEST_DATA_DIR) if f.endswith('.wav')]
        # Limit to 400 positive samples
        wav_files = wav_files[:400]
        print(f"Found {len(wav_files)} positive test samples (limited to 400)")
        
        for i, file in enumerate(wav_files):
            try:
                file_path = os.path.join(TEST_DATA_DIR, file)
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Predict
                is_keyword, confidence = self.predict_sample(audio)
                
                # Record results
                self.results['true_labels'].append(1)  # Positive sample
                self.results['predictions'].append(1 if is_keyword else 0)
                self.results['confidences'].append(confidence)
                
                if is_keyword:
                    self.results['true_positives'] += 1
                else:
                    self.results['false_negatives'] += 1
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(wav_files)} positive samples")
                    
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
        
        return len(wav_files)
    
    def test_negative_samples(self, target_count: int | None = None):
        """Test the model on negative samples (background noise).
        If target_count is provided, test exactly that many negatives (with replacement if needed)."""
        print("🔍 Testing negative samples (background noise)...")
        
        if not os.path.exists(BACKGROUND_NOISE_DIR):
            print(f"❌ Background noise directory not found: {BACKGROUND_NOISE_DIR}")
            return 0
        
        wav_files = [f for f in os.listdir(BACKGROUND_NOISE_DIR) if f.endswith('.wav')]
        if target_count is not None and target_count > 0:
            # Select exactly target_count files; sample with replacement if needed
            idx = np.random.choice(len(wav_files), size=target_count, replace=(len(wav_files) < target_count))
            wav_files = [wav_files[i] for i in idx]
        print(f"Testing {len(wav_files)} negative samples")
        
        for i, file in enumerate(wav_files):
            try:
                file_path = os.path.join(BACKGROUND_NOISE_DIR, file)
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                
                # Predict
                is_keyword, confidence = self.predict_sample(audio)
                
                # Record results
                self.results['true_labels'].append(0)  # Negative sample
                self.results['predictions'].append(1 if is_keyword else 0)
                self.results['confidences'].append(confidence)
                
                if is_keyword:
                    self.results['false_positives'] += 1
                else:
                    self.results['true_negatives'] += 1
                
                if (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(wav_files)} negative samples")
                    
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
        
        return len(wav_files)
    
    def calculate_metrics(self):
        """Calculate comprehensive accuracy metrics"""
        print("\n📊 Calculating Accuracy Metrics...")
        
        # Basic metrics
        total_samples = len(self.results['true_labels'])
        accuracy = accuracy_score(self.results['true_labels'], self.results['predictions'])
        precision = precision_score(self.results['true_labels'], self.results['predictions'])
        recall = recall_score(self.results['true_labels'], self.results['predictions'])
        f1 = f1_score(self.results['true_labels'], self.results['predictions'])
        
        # Confusion matrix
        cm = confusion_matrix(self.results['true_labels'], self.results['predictions'])
        
        # Detailed metrics
        tp = self.results['true_positives']
        fp = self.results['false_positives']
        tn = self.results['true_negatives']
        fn = self.results['false_negatives']
        
        # Print results
        print(f"\n🎯 ACCURACY TEST RESULTS")
        print(f"=" * 50)
        print(f"Total samples tested: {total_samples}")
        print(f"Positive samples: {tp + fn}")
        print(f"Negative samples: {tn + fp}")
        print(f"\n📈 METRICS:")
        print(f"   Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"   Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"   F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        print(f"\n🔍 DETAILED BREAKDOWN:")
        print(f"   True Positives (TP): {tp}")
        print(f"   False Positives (FP): {fp}")
        print(f"   True Negatives (TN): {tn}")
        print(f"   False Negatives (FN): {fn}")
        print(f"\n📊 CONFUSION MATRIX:")
        print(f"   Predicted:    0    1")
        print(f"Actual 0:    {tn:4d} {fp:4d}")
        print(f"Actual 1:    {fn:4d} {tp:4d}")
        
        # Confidence statistics
        confidences = np.array(self.results['confidences'])
        print(f"\n📊 CONFIDENCE STATISTICS:")
        print(f"   Average confidence: {np.mean(confidences):.4f}")
        print(f"   Min confidence: {np.min(confidences):.4f}")
        print(f"   Max confidence: {np.max(confidences):.4f}")
        print(f"   Std confidence: {np.std(confidences):.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'total_samples': total_samples
        }
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("📊 Confusion matrix saved as 'confusion_matrix.png'")
    
    def run_automatic_test(self):
        """Run the complete automatic accuracy test"""
        print("🚀 Starting Automatic Accuracy Test")
        print("=" * 50)
        
        start_time = time.time()
        
        # Test positive samples
        positive_count = self.test_positive_samples()
        
        # Test negative samples (match positive count)
        negative_count = self.test_negative_samples(target_count=positive_count)
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Plot confusion matrix
        self.plot_confusion_matrix(metrics['confusion_matrix'])
        
        end_time = time.time()
        print(f"\n⏱️  Total test time: {end_time - start_time:.2f} seconds")
        print(f"✅ Automatic accuracy test completed!")
        
        return metrics

def main():
    print("🎯 Automatic Accuracy Testing Tool")
    print("=" * 40)
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file not found: {MODEL_PATH}")
        print("Please train the model first using the pipeline")
        return
    
    # Initialize tester
    tester = AutomaticAccuracyTester(MODEL_PATH)
    
    # Ask for detection threshold
    try:
        threshold = float(input(f"Enter detection threshold (0.0-1.0, default {tester.detection_threshold}): ") or tester.detection_threshold)
        tester.detection_threshold = max(0.0, min(1.0, threshold))
    except ValueError:
        print(f"Using default threshold: {tester.detection_threshold}")
    
    # Run automatic test
    results = tester.run_automatic_test()
    
    # Save results to file
    with open('accuracy_test_results.txt', 'w') as f:
        f.write("AUTOMATIC ACCURACY TEST RESULTS\n")
        f.write("=" * 50 + "\n")
        f.write(f"Detection Threshold: {tester.detection_threshold}\n")
        f.write(f"Overall Accuracy: {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {results['precision']:.4f} ({results['precision']*100:.2f}%)\n")
        f.write(f"Recall: {results['recall']:.4f} ({results['recall']*100:.2f}%)\n")
        f.write(f"F1-Score: {results['f1']:.4f} ({results['f1']*100:.2f}%)\n")
        f.write(f"Total Samples: {results['total_samples']}\n")
    
    print(f"\n💾 Results saved to 'accuracy_test_results.txt'")

if __name__ == "__main__":
    main()
