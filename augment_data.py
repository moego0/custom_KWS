import os
import random
import numpy as np
import shutil
import librosa
from scipy.signal import lfilter, butter
import pandas as pd
import soundfile as sf
from audiomentations import Compose, AddGaussianNoise, AddBackgroundNoise
import sounddevice as sd
 
# --- Constants and Configuration ---
MAX_FRAMES = 49
AUG_PER_TECHNIQUE = 5
RECORDINGS_DIR = "recorded_samples"
BACKGROUND_NOISE_DIR = "background_noise"
NEGATIVE_DIR = "negative_samples"
AUGMENTED_DIR = "augmented_data"
ESC50_DIR = "ESC-50"
SAMPLE_RATE = 16000
DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
NUM_MFCC = 40
N_FFT = 2048
HOP_LENGTH = 512

csv_path = "ESC-50/meta/esc50.csv"
audio_path = "ESC-50/audio"

# --- Utility Functions ---
def save_audio_file(file_path, audio_data, sr):
    """Save augmented audio data to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    sf.write(file_path, audio_data, sr)

def extract_mfcc(file_path, sr=SAMPLE_RATE, n_mfcc=NUM_MFCC):
    """
    Extract MFCC features from an audio file.
    Ensures the MFCC shape is consistent for machine learning models.
    """
    try:
        y, _ = librosa.load(file_path, sr=sr)
        y = librosa.util.fix_length(y, size=NUM_SAMPLES)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # Ensure consistent shape (44, 40) - matching the model's expected input
        target_frames = 44  # Fixed target frames to match model input
        if mfcc.shape[1] < target_frames:
            mfcc = np.pad(mfcc, ((0, 0), (0, target_frames - mfcc.shape[1])), mode='constant')
        elif mfcc.shape[1] > target_frames:
            mfcc = mfcc[:, :target_frames]

        return mfcc.T  # Shape: (44, 40)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# --- Core Augmentation Functions ---
def _pitch_shift(audio, sr, n_steps):
    return librosa.effects.pitch_shift(y=audio, sr=sr, n_steps=n_steps)

def _time_stretch(audio, rate):
    return librosa.effects.time_stretch(y=audio, rate=rate)

def _add_noise(audio, noise_factor):
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def _butter_bandpass(audio, sr, lowcut, highcut, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band', analog=False)
    return lfilter(b, a, audio)


from scipy.signal import fftconvolve
import glob
import os

# Load RIR files once
RIR_PATH = "data/rir/"
rir_files = glob.glob(os.path.join(RIR_PATH, "*.wav"))

def _reverb(audio, sr, decay=0.3, use_real_rir=True, rir_prob=0.5):
    """
    Enhanced reverb:
    - If real RIRs available & random chance passes, apply convolution with RIR
    - Else synthetic: multiple reflections with exponential decay
    """
    if use_real_rir and rir_files and random.random() < rir_prob:
        # --- Real RIR convolution ---
        rir_file = random.choice(rir_files)
        rir, _ = librosa.load(rir_file, sr=sr)
        rir = rir / (np.max(np.abs(rir)) + 1e-8)  # normalize
        augmented = fftconvolve(audio, rir, mode='full')[:len(audio)]

    else:
        # --- Synthetic multi-reflection reverb ---
        ir = np.zeros(sr)  # 1 second impulse response
        ir[0] = 1  # dry signal
        # Add 3 random reflections at 50–300 ms with decreasing amplitudes
        for delay_ms in [random.randint(50, 300) for _ in range(3)]:
            delay_samples = int(sr * delay_ms / 1000)
            if delay_samples < sr:
                ir[delay_samples] += decay * (0.5 + 0.5 * random.random())
        augmented = np.convolve(audio, ir, 'full')[:len(audio)]

    # Normalize to prevent clipping
    max_val = np.max(np.abs(augmented))
    if max_val > 0:
        augmented = augmented / max_val

    return augmented


def _echo(audio, sr, delay_s, gain_in, gain_out):
    # Simulates a simple echo effect
    delay_samples = int(delay_s * sr)
    echoed_audio = np.zeros_like(audio)
    echoed_audio += audio * gain_in
    echoed_audio[delay_samples:] += audio[:-delay_samples] * gain_out
    return echoed_audio
def _noise_burst(audio, sr, burst_ms=45):
    """
    Adds a short, loud noise burst (white noise or beep) somewhere in the audio.
    burst_ms: duration of the burst in milliseconds (default 45 ms)
    """
    burst_len = int(sr * burst_ms / 1000)  # samples in burst
    start_idx = random.randint(0, max(0, len(audio) - burst_len))

    # Random choice: white noise or sine beep
    if random.random() < 0.5:
        # White noise burst
        burst = np.random.uniform(-1, 1, burst_len)
    else:
        # Sine beep burst at random frequency
        freq = random.choice([500, 1000, 2000])  # Hz
        t = np.linspace(0, burst_ms / 1000, burst_len, endpoint=False)
        burst = 0.9 * np.sin(2 * np.pi * freq * t)

    # Random amplitude scaling (so not always max loudness)
    burst *= random.uniform(0.5, 1.0)

    # Make a copy to avoid modifying original
    augmented = np.copy(audio)

    # If burst is longer than available audio, trim it
    end_idx = min(start_idx + burst_len, len(audio))
    burst = burst[:end_idx - start_idx]

    # Add burst into audio
    augmented[start_idx:end_idx] += burst

    # Normalize to avoid clipping
    max_val = np.max(np.abs(augmented))
    if max_val > 0:
        augmented = augmented / max_val
    return augmented

def _shift_time(audio, sr, shift_amount):
    shift_size = int(sr * shift_amount)
    return np.roll(audio, shift_size)

def _random_crop(audio, sr, crop_size):
    if len(audio) <= crop_size:
        return audio
    start_index = random.randint(0, len(audio) - crop_size)
    return audio[start_index:start_index + crop_size]
    
def _dropout_segments(audio, dropout_fraction):
    segment_size = int(len(audio) * dropout_fraction)
    start_index = random.randint(0, len(audio) - segment_size)
    augmented_audio = np.copy(audio)
    augmented_audio[start_index:start_index + segment_size] = 0
    return augmented_audio

def _gain(audio, gain_db):
    gain_linear = 10**(gain_db / 20)
    return audio * gain_linear

def _low_pass(audio, sr, highcut, order=5):
    nyq = 0.5 * sr
    high = highcut / nyq
    b, a = butter(order, high, btype='low', analog=False)
    return lfilter(b, a, audio)
    
def _high_pass(audio, sr, lowcut, order=5):
    nyq = 0.5 * sr
    low = lowcut / nyq
    b, a = butter(order, low, btype='high', analog=False)
    return lfilter(b, a, audio)

# --- Augmentation Loop ---
def augment_audio(input_dir, output_dir):
    """
    Loads original audio files, applies a diverse set of augmentations,
    and saves the augmented files to a new directory.
    """
    techniques = [
        'pitch_shift', 'time_stretch', 'add_noise', 'bandpass', 'reverb', 'echo',
        'gain', 'low_pass', 'high_pass', 'shift_time', 'random_crop', 'dropout_segments', 'noise_burst'
    ]
    
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".wav"):
            original_file_path = os.path.join(input_dir, filename)
            audio, sr = librosa.load(original_file_path, sr=SAMPLE_RATE)

            # Ensure consistent length before augmentation
            audio = librosa.util.fix_length(audio, size=NUM_SAMPLES)

            for tech in techniques:
                for i in range(AUG_PER_TECHNIQUE):
                    augmented_audio = audio
                    
                    if tech == 'pitch_shift':
                        n_steps = random.uniform(-4, 4)
                        augmented_audio = _pitch_shift(audio, sr, n_steps)
                    elif tech == 'time_stretch':
                        rate = random.uniform(0.8, 1.25)
                        augmented_audio = _time_stretch(audio, rate)
                    elif tech == 'add_noise':
                        noise_factor = random.uniform(0.005, 0.05)
                        augmented_audio = _add_noise(audio, noise_factor)
                    elif tech == 'bandpass':
                        lowcut = random.uniform(50, 400)
                        highcut = random.uniform(3000, 7000)
                        augmented_audio = _butter_bandpass(audio, sr, lowcut, highcut)
                    elif tech == 'reverb':
                        decay = random.uniform(0.1, 0.5)
                        augmented_audio = _reverb(audio, sr, decay)
                    elif tech == 'echo':
                        delay_s = random.uniform(0.01, 0.5)
                        gain_in = random.uniform(0.5, 1.0)
                        gain_out = random.uniform(0.1, 0.5)
                        augmented_audio = _echo(audio, sr, delay_s, gain_in, gain_out)
                    elif tech == 'gain':
                        gain_db = random.uniform(-10, 10)
                        augmented_audio = _gain(audio, gain_db)
                    elif tech == 'low_pass':
                        highcut = random.uniform(2000, 7000)
                        augmented_audio = _low_pass(audio, sr, highcut)
                    elif tech == 'high_pass':
                        lowcut = random.uniform(50, 400)
                        augmented_audio = _high_pass(audio, sr, lowcut)
                    elif tech == 'shift_time':
                        shift_amount = random.uniform(-0.5, 0.5)
                        augmented_audio = _shift_time(audio, sr, shift_amount)
                    elif tech == 'random_crop':
                        crop_size = int(NUM_SAMPLES * random.uniform(0.7, 1.0))
                        cropped_audio = _random_crop(audio, sr, crop_size)
                        # Pad the cropped audio back to the original size
                        augmented_audio = librosa.util.fix_length(cropped_audio, size=NUM_SAMPLES)
                    elif tech == 'noise_burst':
                        augmented_audio = _noise_burst(audio, sr)
                    elif tech == 'dropout_segments':
                        dropout_fraction = random.uniform(0.1, 0.5)
                        augmented_audio = _dropout_segments(audio, dropout_fraction)
                        # Pad if dropout makes it shorter (it shouldn't, but as a safeguard)
                        augmented_audio = librosa.util.fix_length(augmented_audio, size=NUM_SAMPLES)

                    output_file_name = f"{os.path.splitext(filename)[0]}_{tech}_{i+1}.wav"
                    output_file_path = os.path.join(output_dir, output_file_name)
                    save_audio_file(output_file_path, augmented_audio, sr)

def augment_audio_default():
    """
    Wrapper function that calls augment_audio with default directories.
    """
    augment_audio(RECORDINGS_DIR, AUGMENTED_DIR)
    return True
            
# --- Other Original Functions (kept for compatibility) ---
def record_samples(label_name, record_path=RECORDINGS_DIR, duration=DURATION, sr=SAMPLE_RATE):
    """
    Records multiple audio samples from the user for a given keyword.
    """
    os.makedirs(record_path, exist_ok=True)
    print(f"Recording {AUG_PER_TECHNIQUE} samples for keyword: '{label_name}'")
    for i in range(AUG_PER_TECHNIQUE):
        print(f"Press Enter to record sample {i+1}...")
        input()
        print("🎙️ Recording...")
        
        try:
            audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, blocking=True)
            audio = audio.flatten()
            
            file_path = os.path.join(record_path, f"{label_name}_{i}.wav")
            sf.write(file_path, audio, sr)
            print(f"💾 Saved: {file_path}")
            
        except KeyboardInterrupt:
            print("\n❌ Recording interrupted by user.")
            return False
        except Exception as e:
            print(f"❌ Error during recording: {e}")
            return False
    
    print("✅ Recording completed successfully!")
    return True

def prepare_negative_samples(esc50_dir, output_dir, count=200):
    """
    Extracts a specified number of negative samples from the ESC-50 dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(os.path.join(esc50_dir, 'meta/esc50.csv'))
    
    # Filter for non-speech sounds (assuming we're only interested in speech)
    non_speech_categories = df[~df['category'].isin(['coughing', 'speech', 'sneeze'])]
    negative_files = non_speech_categories['filename'].tolist()
    
    if len(negative_files) < count:
        print(f"Warning: Only {len(negative_files)} negative samples available. Using all of them.")
        samples_to_use = negative_files
    else:
        samples_to_use = random.sample(negative_files, count)
        
    for i, file_name in enumerate(samples_to_use):
        src = os.path.join(esc50_dir, 'audio', file_name)
        dst = os.path.join(output_dir, f'negative_{i}.wav')
        shutil.copy(src, dst)
        
    print(f"✅ Prepared {len(samples_to_use)} negative samples.")
    return True

def _count_augmented_wavs() -> int:
    try:
        if not os.path.exists(AUGMENTED_DIR):
            return 0
        return sum(1 for f in os.listdir(AUGMENTED_DIR) if f.endswith('.wav'))
    except Exception:
        return 0

def prepare_negative_samples_default():
    """
    Prepare negatives from ESC-50 matching augmented wav count when available; fallback to 300.
    """
    count = _count_augmented_wavs() or 300
    return prepare_negative_samples(ESC50_DIR, NEGATIVE_DIR, count)

def get_features_for_directory(dir_path):
    """
    Extracts MFCC features from all .wav files in a directory.
    """
    mfccs = []
    labels = []
    
    for file_name in os.listdir(dir_path):
        if file_name.endswith(".wav"):
            file_path = os.path.join(dir_path, file_name)
            
            # Assuming file names are in the format "keyword_*.wav"
            label = file_name.split('_')[0]
            
            mfcc = extract_mfcc(file_path)
            if mfcc is not None:
                mfccs.append(mfcc)
                labels.append(label)
    
    if mfccs:
        return np.array(mfccs), np.array(labels)
    else:
        return np.array([]), np.array([])
        
def collect_all_augmented_samples():
    """Collect all augmented sample file paths."""
    positive_samples = []
    
    if os.path.exists(AUGMENTED_DIR):
        for file_name in os.listdir(AUGMENTED_DIR):
            if file_name.endswith('.wav'):
                file_path = os.path.join(AUGMENTED_DIR, file_name)
                positive_samples.append(file_path)
    
    return positive_samples

def preprocess_esc50_to_fixed_length(audio_path='ESC-50/audio', output_path='processed_esc50', target_sr=16000, duration_sec=1):
    """
    Preprocess ESC-50 audio files to fixed length (1 second) and target sample rate.
    """
    os.makedirs(output_path, exist_ok=True)
    target_len = target_sr * duration_sec

    for file in os.listdir(audio_path):
        if file.endswith(".wav"):
            y, sr = librosa.load(os.path.join(audio_path, file), sr=target_sr, mono=True)
            if len(y) > target_len:
                y = y[:target_len]
            else:
                y = librosa.util.fix_length(y, size=target_len)
            sf.write(os.path.join(output_path, file), y, target_sr)

    print(f"✅ All ESC-50 files have been preprocessed into 1s, 16kHz mono format in '{output_path}'.")

if __name__ == '__main__':
    # Define directories
    original_data_dir = 'data/original'
    augmented_data_dir = 'data/augmented'
    negative_samples_dir = 'data/negative'

    # Example Usage:
    # 1. Record original samples (uncomment to use)
    # record_samples(label_name='your_keyword')

    # 2. Augment the data
    print(f"Starting data augmentation for files in  '{original_data_dir}'...")
    augment_audio(original_data_dir, augmented_data_dir)
    print("Data augmentation complete.")

    # 3. Prepare negative samples (uncomment and provide path to ESC-50 dataset)
    # print(f"Preparing negative samples from ESC-50 dataset...")
    # prepare_negative_samples(esc50_dir='path/to/ESC-50', output_dir=negative_samples_dir)
    # print("Negative samples prepared.")

    # 4. Extract features for ML model
    print("Extracting features from augmented data...")
    augmented_features, augmented_labels = get_features_for_directory(augmented_data_dir)
    print(f"Extracted features shape: {augmented_features.shape}")
