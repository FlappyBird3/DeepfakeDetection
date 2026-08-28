"""
preprocessing.py

This file contains the EXACT SAME audio -> spectrogram logic used during
training (from your MyDataset.__getitem__ and resample_audio functions).

WHY THIS MUST MATCH TRAINING EXACTLY:
Your model learned patterns based on spectrograms shaped a very specific
way (16kHz, 64000 samples, CQT with n_bins=84, hop_length=512,
bins_per_octave=12). If preprocessing here differs even slightly, the
model will receive input that looks nothing like what it trained on,
and predictions will be meaningless (garbage in, garbage out).
"""

import librosa
import numpy as np
import torch

SAMPLE_RATE = 16000
TARGET_SAMPLES = 64000


def resample_audio(file_path):
    """
    Loads an audio file, resamples to 16kHz, and forces it to exactly
    64000 samples (padding short clips with silence, cropping long ones).
    Returns None if loading fails for any reason.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        if len(y) < TARGET_SAMPLES:
            extra_padding = TARGET_SAMPLES - len(y)
            y = np.pad(y, (0, extra_padding), 'constant')
        else:
            y = y[0:TARGET_SAMPLES]
        return y
    except Exception as e:
        print(f"Error occurred while loading audio file: {e}")
        return None


def audio_file_to_model_input(file_path):
    """
    Full pipeline: file path -> ready-to-use tensor for the model.

    Returns: a torch.Tensor of shape (1, 1, 84, num_time_frames)
             (batch dimension of 1, channel dimension of 1 - same
             .unsqueeze(1) trick from training, done here manually
             since we only ever have ONE sample at a time in the API,
             not a batch of 32 like during training)

    Returns None if audio loading failed.
    """
    audio_data = resample_audio(file_path)
    if audio_data is None:
        return None

    matrix = np.abs(librosa.cqt(
        audio_data,
        sr=SAMPLE_RATE,
        hop_length=512,
        n_bins=84,
        bins_per_octave=12,
    ))
    decibel_matrix = librosa.amplitude_to_db(matrix, ref=np.max)

    tensor = torch.tensor(decibel_matrix, dtype=torch.float32)
    # Add batch dimension (1 sample) AND channel dimension (1 channel),
    # matching the shape the model's conv1 layer expects: (batch, channel, height, width)
    tensor = tensor.unsqueeze(0).unsqueeze(0)
    return tensor
