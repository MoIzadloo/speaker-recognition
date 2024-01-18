import librosa
import os
import numpy as np
import pandas as pd

def extract_mfccs(audio_file_path):
    """
    Extracts MFCCs from an audio file and returns a list of MFCC vectors
    """
    # Load the audio file
    audio, sr = librosa.load(audio_file_path)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)

    # Convert MFCCs to a list of vectors
    mfcc_list = mfccs.transpose().tolist()

    return mfcc_list

def generate_mfccs_dataset(audio_dir, save_path):
    """
    Generates a dataset of MFCCs from a directory of audio files and saves it to a CSV file
    """
    # Initialize an empty list to store extracted MFCCs
    mfcc_data = []

    # Iterate through each audio file in the directory
    for audio_file in os.listdir(audio_dir):
        if audio_file.endswith('.mp3') or audio_file.endswith('.ogg'):
            # Extract MFCCs from the audio file
            mfcc_vectors = extract_mfccs(os.path.join(audio_dir, audio_file))

            # Create a dictionary to store the MFCCs and the corresponding speaker ID
            mfcc_data.append({
                'mfcc': mfcc_vectors,
                'speaker_id': int(audio_file.split('_')[0])
            })

    # Save the MFCC dataset to a CSV file
    df = pd.DataFrame(mfcc_data)
    df.to_csv(save_path, index=False)

# Specify the directory containing the audio files
samples_audio_dir = 'samples'

# Specify the path to save the MFCCs dataset
samples_save_path = 'samples_dataset.csv'

# Generate the MFCCs dataset
generate_mfccs_dataset(samples_audio_dir, samples_save_path)

# Specify the directory containing the audio files
samples_audio_dir = 'tests'

# Specify the path to save the MFCCs dataset
samples_save_path = 'tests_dataset.csv'

# Generate the MFCCs dataset
generate_mfccs_dataset(samples_audio_dir, samples_save_path)