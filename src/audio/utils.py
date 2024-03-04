import numpy as np

def normalize_audio(audio_data):
    """
    Normalize the audio data to the range [-1, 1].

    :param audio_data: NumPy array of audio data.
    :return: Normalized audio data.
    """
    # Find the absolute maximum value in the audio data
    max_val = np.abs(audio_data).max()
    # Avoid division by zero
    if max_val == 0:
        return audio_data
    # Normalize the audio data
    return audio_data / max_val