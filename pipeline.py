import os
import numpy as np
from sklearn.utils import shuffle
from agument_data import collect_all_augmented_samples, augment_audio_default, prepare_negative_samples_default, preprocess_esc50_to_fixed_length, extract_mfcc
from bulid_model import train_and_export
from agument_data import *

RECORDINGS_DIR = "recorded_samples"
AUGMENTED_DIR = "augmented_data"
NEGATIVE_DIR = "negative_samples"
ESC50_DIR = "ESC-50"
ESC50_TRAIN_DIR = "processed_esc50"
ESC50_TEST_DIR = "processed_esc50_test"

def check_pipeline_status():
    """Check the status of all pipeline components"""
    status = {
        'recordings': os.path.exists(RECORDINGS_DIR) and len(os.listdir(RECORDINGS_DIR)) > 0,
        'augmented': os.path.exists(AUGMENTED_DIR) and len(os.listdir(AUGMENTED_DIR)) > 0,
        'negative': os.path.exists(NEGATIVE_DIR) and len(os.listdir(NEGATIVE_DIR)) > 0,
        'processed_esc50': os.path.exists("processed_esc50") and len(os.listdir("processed_esc50")) > 0
    }
    
    print("\n📊 Pipeline Status:")
    print(f"   📝 Recorded samples: {'✅' if status['recordings'] else '❌'}")
    print(f"   🔄 Augmented samples: {'✅' if status['augmented'] else '❌'}")
    print(f"   🎵 Negative samples: {'✅' if status['negative'] else '❌'}")
    print(f"   🔧 Processed ESC-50: {'✅' if status['processed_esc50'] else '❌'}")
    
    return status

def ask_user_choice(prompt, default_yes=True):
    """Ask user for yes/no choice with a default"""
    while True:
        response = input(f"{prompt} ({'Y/n' if default_yes else 'y/N'}): ").lower().strip()
        if response == '':
            return default_yes
        elif response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please enter 'y' or 'n'")

if __name__ == "__main__":
    print("🚀 Starting Keyword Spotting Pipeline...")
    
    # Check initial status
    initial_status = check_pipeline_status()
    
    # Step 1: Check if recorded samples exist, if not start recording
    if not initial_status['recordings']:
        print("\n📝 No recorded samples found.")
        if ask_user_choice("Do you want to record new samples"):
            if not record_samples("keyword"):
                print("❌ Recording failed. Exiting.")
                exit(1)
        else:
            print("❌ Cannot proceed without recorded samples. Exiting.")
            exit(1)
    else:
        print(f"\n✅ Found {len(os.listdir(RECORDINGS_DIR))} recorded samples in {RECORDINGS_DIR}")
        if ask_user_choice("Do you want to record new samples (this will replace existing ones)", default_yes=False):
            if not record_samples("keyword"):
                print("❌ Recording failed. Exiting.")
                exit(1)
    
    # Step 2: Check if augmented samples exist, if not start augmentation
    if not initial_status['augmented']:
        print("\n🔄 No augmented samples found.")
        if ask_user_choice("Do you want to create augmented samples"):
            if not augment_audio_default():
                print("❌ Augmentation failed. Exiting.")
                exit(1)
        else:
            print("⚠️  Proceeding without augmented samples (will use original recordings)")
    else:
        print(f"\n✅ Found augmented samples in {AUGMENTED_DIR}")
        if ask_user_choice("Do you want to recreate augmented samples", default_yes=False):
            if not augment_audio_default():
                print("❌ Augmentation failed. Exiting.")
                exit(1)
            else:
                print("✅ Augmented samples recreated")
                aug_count = sum(1 for f in os.listdir(AUGMENTED_DIR) if f.endswith('.wav'))
                print(f"Number of augmented samples: {aug_count}")
    # Step 3: Generate negative samples from ESC-50
    if not initial_status['negative']:
        print("\n🎵 No negative samples found.")
        if ask_user_choice("Do you want to generate negative samples from ESC-50"):
            prepare_negative_samples_default()
        else:
            print("⚠️  Proceeding without negative samples")
    else:
        print(f"\n✅ Found {len(os.listdir(NEGATIVE_DIR))} negative samples")
        if ask_user_choice("Do you want to regenerate negative samples", default_yes=False):
            prepare_negative_samples_default()
    
    # Step 4: Preprocess ESC-50 to fixed length
    if not initial_status['processed_esc50']:
        print("\n🔧 ESC-50 not preprocessed.")
        if ask_user_choice("Do you want to preprocess ESC-50"):
            preprocess_esc50_to_fixed_length()
        else:
            print("⚠️  Proceeding without processed ESC-50")
    else:
        print("\n✅ ESC-50 preprocessing completed")
        if ask_user_choice("Do you want to reprocess ESC-50", default_yes=False):
            preprocess_esc50_to_fixed_length()

    # Final status check
    final_status = check_pipeline_status()
    
    print("\n--- Step 5: Extract Features ---")
    
    # Collect positive samples (augmented recordings)
    positive_samples = collect_all_augmented_samples()
    
    # If no augmented samples, use recorded samples directly
    if not positive_samples and os.path.exists(RECORDINGS_DIR):
        positive_samples = [os.path.join(RECORDINGS_DIR, f) for f in os.listdir(RECORDINGS_DIR) if f.endswith('.wav')]
    
    # Negative samples from processed ESC-50
    negative_path = "processed_esc50"
    
    if not positive_samples:
        print("❌ No positive samples found. Please record samples first.")
        exit(1)
    
    if not os.path.exists(negative_path):
        print("❌ Negative samples directory not found. Please run preprocessing first.")
        exit(1)

    # Extract MFCC features
    print("Extracting MFCC features from positive samples...")
    X_pos = [extract_mfcc(file_path) for file_path in positive_samples]
    X_pos = [x for x in X_pos if x is not None]  # Filter out None values
    
    print("Extracting MFCC features from negative samples...")
    # Balance negatives to match number of positives
    negative_files = [file for file in os.listdir(negative_path) if file.endswith(".wav")]
    if not negative_files:
        print("❌ No negative samples found in processed_esc50. Exiting.")
        exit(1)

    num_pos = len(X_pos)
    if num_pos == 0:
        print("❌ No positive samples available after feature extraction. Exiting.")
        exit(1)

    # Sample with replacement if needed to ensure equal counts
    indices = np.random.choice(len(negative_files), size=num_pos, replace=(len(negative_files) < num_pos))
    selected_negatives = [negative_files[i] for i in indices]
    X_neg = [extract_mfcc(os.path.join(negative_path, file)) for file in selected_negatives]
    X_neg = [x for x in X_neg if x is not None]  # Filter out None values

    # Labels
    y_pos = [1] * len(X_pos)
    y_neg = [0] * len(X_neg)

    # Combine and shuffle
    X = np.array(X_pos + X_neg)
    y = np.array(y_pos + y_neg)
    X, y = shuffle(X, y, random_state=42)

    # Expand dims to match CNN input shape
    X = X[..., np.newaxis]

    print(f"Dataset shape: {X.shape}")
    print(f"Positive samples: {len(X_pos)}")
    print(f"Negative samples (balanced): {len(X_neg)}")

    print("\n--- Step 6: Train and Export Model ---")
    train_and_export(X, y)

