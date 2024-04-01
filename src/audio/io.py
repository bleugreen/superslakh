import librosa
import pretty_midi
import torch
import subprocess
import os
import numpy as np
from src.audio.utils import normalize_audio

def load_audio_segment_beats(audio_filepath, downbeats, beats, sample_rate):
    """
    Loads audio segment from the first downbeat to the end
    """
    duration = max(downbeats[-1], beats[-1]) - downbeats[0]
    audio, _ = librosa.load(audio_filepath, sr=sample_rate, offset=downbeats[0], duration=duration)

    adjusted_downbeats = [db - downbeats[0] for db in downbeats]
    last_db_interval = downbeats[-1] - downbeats[-2]
    adjusted_downbeats.append(downbeats[-1]+last_db_interval)
    adjusted_beats = [b - downbeats[0] for b in beats]
    last_b_interval = beats[-1] - beats[-2]
    adjusted_downbeats.append(beats[-1]+last_b_interval)
    return torch.Tensor(audio), adjusted_downbeats, adjusted_beats

def save_audio(audio_data, output_path, normalize=False, sr=44100):
    """
    Saves audio data to a specified path in FLAC format.

    :param audio_data: The audio data to be saved.
    :param output_path: The path (including filename) where the audio will be saved.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir != '':
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  
    if normalize:
        audio_data = normalize_audio(audio_data)
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    audio_bytes = audio_data_int16.tobytes()

    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-f', 's16le',  # Input format is signed 16-bit little endian
        '-ar', str(sr),  # Sample rate
        '-ac', '1',  # Number of audio channels
        '-i', 'pipe:0',  # Read input from stdin
        '-compression_level', str(8),  # Max FLAC compression
        output_path  # Output file path
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process.stdin.write(audio_bytes)
    process.stdin.close()
    process.wait()

    if process.returncode != 0:
        raise Exception(f"FFmpeg returned non-zero exit code: {process.returncode}. Check FFmpeg command and inputs.")




def split_audio_by_tempo(audio_path, midi_path, sample_rate=22050):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sample_rate)
    # Load the MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_path)
    # Get the tempo changes from the MIDI file
    times, tempos = midi_data.get_tempo_changes()
    # Initialize variables
    audio_segments = []
    prev_time = 0
    # Iterate over the tempo changes
    for time, tempo in zip(times, tempos):
        # Convert MIDI time to audio samples
        start_sample = int(prev_time * sr)
        end_sample = int(time * sr)
        # Extract the audio segment
        audio_segment = y[start_sample:end_sample]
        # Calculate the BPM
        bpm = int(tempo)
        # Append the (bpm, audio) tuple to the list
        if len(audio_segment) > 0:
            audio_segments.append((bpm, audio_segment))
        prev_time = time
    # Extract the final audio segment
    start_sample = int(prev_time * sr)
    audio_segment = y[start_sample:]
    # Get the final tempo
    final_tempo = tempos[-1]
    bpm = int(final_tempo)
    # Append the final (bpm, audio) tuple to the list
    audio_segments.append((bpm, audio_segment))
    return audio_segments