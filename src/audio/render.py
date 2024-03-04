import os
import fluidsynth
import numpy as np
import torch
import torchaudio.transforms as T
from scipy.stats import skewnorm
import torchaudio
from src.audio.utils import normalize_audio
from src.audio.io import save_audio

def generate_loudness():
    alpha = -4
    loc = -12
    scale = 3
    upper_bound = -5
    value = skewnorm.rvs(alpha, loc, scale)
    return min(value, upper_bound)

def adjust_volume(audio, sr, target_loudness):
    loudness = torchaudio.functional.loudness(audio, sample_rate=sr)
    # Convert loudness (LKFS) to dB
    gain = target_loudness-loudness
    vol_transform = T.Vol(gain=gain, gain_type='db')
    return vol_transform(audio)

def mix_audios(audio_paths, output_path="mixed_audio.flac"):
    audios = []
    lengths = []
    target_loudness = generate_loudness()
    for path in audio_paths:
        audio, sr = torchaudio.load(path)
        audios.append(normalize_audio(audio.cpu()))
        lengths.append(audio.shape[1])

    max_length = max(lengths)
    mixed_audio = torch.zeros(1, max_length)
    for audio in audios:
        padded_audio = torch.nn.functional.pad(audio, (0, max_length - audio.shape[1]))
        mixed_audio += padded_audio

    mixed_audio = adjust_volume(mixed_audio, sr, target_loudness)
    save_audio(mixed_audio.numpy(), output_path)
    return output_path


def make_synth(sf2_path, sr=44100, prefix='/media/bleu/bulkdata2/superslakh/soundfonts'):
    full_path = os.path.join(prefix, sf2_path)
    synthesizer = fluidsynth.Synth(samplerate=float(sr))
    sfid = synthesizer.sfload(full_path)
    return synthesizer, sfid

def fluidsynthesize(instrument, fs=44100, synthesizer=None, sfid=0, channel=0):
    """
        Adapted from pretty_midi.synthesize()
    """
    
    # If the instrument has no notes, return an empty array
    if len(instrument.notes) == 0:
        print('empty stem')
        return np.array([])

    # Collect all notes in one list
    event_list = []
    for note in instrument.notes:
        event_list += [[note.start, 'note on', note.pitch, note.velocity]]
        event_list += [[note.end, 'note off', note.pitch]]
    for bend in instrument.pitch_bends:
        event_list += [[bend.time, 'pitch bend', bend.pitch]]
    for control_change in instrument.control_changes:
        event_list += [[control_change.time, 'control change',
                        control_change.number, control_change.value]]
    # Sort the event list by time, and secondarily by whether the event
    # is a note off
    event_list.sort(key=lambda x: (x[0], x[1] != 'note off'))
    # Add some silence at the beginning according to the time of the first
    # event
    current_time = event_list[0][0]
    # Convert absolute seconds to relative samples
    next_event_times = [e[0] for e in event_list[1:]]
    for event, end in zip(event_list[:-1], next_event_times):
        event[0] = end - event[0]
    # Include 1 second of silence at the end
    event_list[-1][0] = 1.
    # Pre-allocate output array
    total_time = current_time + np.sum([e[0] for e in event_list])
    synthesized = np.zeros(int(np.ceil(fs*total_time)))
    # Iterate over all events

    for event in event_list:
        # Process events based on type
        if event[1] == 'note on':
            synthesizer.noteon(channel, event[2], event[3])
        elif event[1] == 'note off':
            synthesizer.noteoff(channel, event[2])
        elif event[1] == 'pitch bend':
            synthesizer.pitch_bend(channel, event[2])
        elif event[1] == 'control change':
            synthesizer.cc(channel, event[2], event[3])
        # Add in these samples
        current_sample = int(fs*current_time)
        end = int(fs*(current_time + event[0]))

        samples = synthesizer.get_samples(end - current_sample)[::2]

        synthesized[current_sample:end] += samples
        # Increment the current sample
        current_time += event[0]

    return synthesized