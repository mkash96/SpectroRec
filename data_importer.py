import os
import pandas as pd
import re
import math
import numpy as np
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt

"""
Convert 30s mp3 files into mel-spectrograms.

A mel-spectrograms is a kind of time-frequency representation.
It is obtained from an audio signal by computing the Fourier transforms of short, overlapping windows.
Each of these Fourier transforms constitutes a frame.
These successive frames are then concatenated into a matrix to form the spectrogram.
"""

def create_spectrogram(verbose=0, mode=None):
    if mode == "Train":
        if os.path.exists('Train_Spectogram_Images'):
            return
        # Get Genres and Track IDs from the tracks.csv file
        filename_metadata = "Dataset/fma_metadata/tracks.csv"
        tracks = pd.read_csv(filename_metadata, header=2, low_memory=False)
        tracks_array = tracks.values
        tracks_id_array = tracks_array[:, 0]
        tracks_genre_array = tracks_array[:, 40]
        tracks_id_array = tracks_id_array.reshape(tracks_id_array.shape[0], 1)
        tracks_genre_array = tracks_genre_array.reshape(tracks_genre_array.shape[0], 1)

        # Ensure track IDs are strings
        tracks_id_array = [str(int(x[0])) for x in tracks_id_array]

        #Gets the subdirectories from fma_small folder
        folder_sample = "Dataset/fma_small"
        directories = [d for d in os.listdir(folder_sample)
                       if os.path.isdir(os.path.join(folder_sample, d))]
        counter = 0
        if(verbose > 0):
            print("Converting mp3 audio files into mel Spectograms ...")
        if not os.path.exists('Train_Spectogram_Images'):
            os.makedirs('Train_Spectogram_Images')

        #Iterates over each subdirectory in the sample folder.Collects all MP3 files within each subdirectory.
        for d in directories:
            label_directory = os.path.join(folder_sample, d)
            file_names = [os.path.join(label_directory, f)
                          for f in os.listdir(label_directory)
                          if f.endswith(".mp3")]

            # Convert .mp3 files into mel-Spectograms
            for f in file_names:
                track_id = int(re.search('fma_small/.*/(.+?).mp3', f).group(1))     #extracts track_id from the file path
                print(f"Processing track ID: {track_id}, File: {f}")

                try:
                    track_index = tracks_id_array.index(str(track_id))      #Finds the index of the current track ID in the tracks_id_array.
                except ValueError:
                    print(f"Track ID {track_id} not found in metadata, skipping...")
                    continue

                if str(tracks_genre_array[track_index, 0]) != '0':       #If the genre is not '0', the code inside the if block will execute; otherwise, it will be skipped.
                    print(f"Creating spectrogram for: {f}")
                    try:
                        y, sr = librosa.load(f)     #y: Array containing the audio time series data (the amplitude values of audio signal over time). sr: Sampling rate of audio file (number of samples per second).
                    except Exception as e:
                        print(f"Error loading {f}: {e}")
                        continue  # Skip this file if it fails to load
                    
                    #n_mels Specifies the number of mel bands to generate. Spectrogram will have 128 mel frequency bins.
                    melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)    #fmax Sets the maximum frequency (in Hz) considered in the spectrogram. Frequencies above this value are ignored. The human ear can typically hear up to 20,000 Hz, but many important audio features are below 8000 Hz.
                    mel = librosa.power_to_db(melspectrogram_array)     #This function converts a power spectrogram (amplitude squared) to decibel units.
                    # Length and Width of Spectogram
                    fig_size = plt.rcParams["figure.figsize"]
                    fig_size[0] = float(mel.shape[1]) / float(100)
                    fig_size[1] = float(mel.shape[0]) / float(100)
                    plt.rcParams["figure.figsize"] = fig_size
                    plt.axis('off')
                    plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
                    librosa.display.specshow(mel, cmap='gray_r')
                    plt.savefig(f"Train_Spectogram_Images/{counter}_{str(tracks_genre_array[track_index, 0])}.jpg", bbox_inches=None, pad_inches=0)
                    plt.close()
                    counter += 1

        print(f"Total spectrograms generated: {counter}")
        return



    elif mode == "Test":
        if os.path.exists('Test_Spectogram_Images'):
            return

        folder_sample = "Dataset/DLMusicTest_30"
        counter = 0
        if(verbose > 0):
            print ("Converting mp3 audio files into mel Spectograms ...")
        if not os.path.exists('Test_Sepctogram_Images'):
            os.makedirs('Test_Spectogram_Images')
        file_names = [os.path.join(folder_sample, f) for f in os.listdir(folder_sample)
                       if f.endswith(".mp3")]
        # Convert .mp3 files into mel-Spectograms
        for f in file_names:
            test_id = re.search('Dataset/DLMusicTest_30/(.+?).mp3', f).group(1)

            y, sr = librosa.load(f)
            melspectrogram_array = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)
            mel = librosa.power_to_db(melspectrogram_array)
            # Length and Width of Spectogram
            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = float(mel.shape[1]) / float(100)
            fig_size[1] = float(mel.shape[0]) / float(100)
            plt.rcParams["figure.figsize"] = fig_size
            plt.axis('off')
            plt.axes([0., 0., 1., 1.0], frameon=False, xticks=[], yticks=[])
            librosa.display.specshow(mel, cmap='gray_r')
            plt.savefig("Test_Spectogram_Images/"+test_id+".jpg", bbox_inches=None, pad_inches=0)
            plt.close()
        return
