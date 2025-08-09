#!/usr/bin/env python3
"""
Simple recording script for testing keyword spotting.
Records 1-second audio samples when you press Enter.
Saves recordings in TESTING/test_recordings/ directory.
"""

import os
import sounddevice as sd
import soundfile as sf
import numpy as np
from datetime import datetime

# Configuration
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds
OUTPUT_DIR = "TESTING/test_recordings"

def setup_output_directory():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"📁 Output directory: {os.path.abspath(OUTPUT_DIR)}")

def record_audio():
    """Record 1 second of audio."""
    print("🎙️ Recording... (1 second)")
    try:
        # Record audio
        audio = sd.rec(int(DURATION * SAMPLE_RATE), 
                      samplerate=SAMPLE_RATE, 
                      channels=1, 
                      blocking=True)
        
        # Flatten to 1D array
        audio = audio.flatten()
        
        return audio
    except Exception as e:
        print(f"❌ Error during recording: {e}")
        return None

def save_audio(audio_data):
    """Save audio data to file with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"test_sample_{timestamp}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        sf.write(filepath, audio_data, SAMPLE_RATE)
        print(f"💾 Saved: {filename}")
        return filepath
    except Exception as e:
        print(f"❌ Error saving file: {e}")
        return None

def main():
    """Main recording loop."""
    print("🎤 Audio Recording Tool for Testing")
    print("=" * 40)
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"Duration: {DURATION} seconds")
    print("=" * 40)
    
    # Setup output directory
    setup_output_directory()
    
    print("\nInstructions:")
    print("• Press Enter to start recording")
    print("• Press Ctrl+C to stop and exit")
    print("• Make sure your microphone is working")
    
    recording_count = 0
    
    try:
        while True:
            print(f"\n--- Recording #{recording_count + 1} ---")
            input("Press Enter to record: ")
            
            # Record audio
            audio_data = record_audio()
            
            if audio_data is not None:
                # Save audio
                saved_file = save_audio(audio_data)
                
                if saved_file:
                    recording_count += 1
                    print(f"✅ Recording #{recording_count} completed!")
                    
                    # Show some basic info about the recording
                    max_amplitude = np.max(np.abs(audio_data))
                    print(f"   Max amplitude: {max_amplitude:.4f}")
                    
                    if max_amplitude < 0.01:
                        print("   ⚠️  Warning: Very quiet recording. Check your microphone.")
                else:
                    print("❌ Failed to save recording.")
            else:
                print("❌ Failed to record audio.")
                
    except KeyboardInterrupt:
        print(f"\n\n🛑 Recording stopped by user (Ctrl+C)")
        print(f"\n📊 Summary:")
        print(f"Total recordings: {recording_count}")
        print(f"Files saved in: {os.path.abspath(OUTPUT_DIR)}")
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
