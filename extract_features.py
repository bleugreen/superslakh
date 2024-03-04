from src.batch_proc import BatchProcessor
from src.audio.io import load_audio_segment
from src.audio.filter import create_log_filter, apply_log_filter
from src.midi.calc import calculate_beat_phase
from dataclasses import dataclass
import json
import torch
import torchaudio
import nnAudio.features as feat
import h5py
import multiprocessing
import numpy as np
from src.db import SQLiteClient
import os

NUM_WORKERS = 12
NUM_BATCH = 32
DEVICE = 'cuda'
SAMPLE_RATE = 22050
N_FFT = 2048
HOP = 512
OUTPUT_PATH = '../phasefinder/stft_db_b_phase_big.hdf5'
GROUP = 'data'

@dataclass
class TaskItem:
    id: int
    filename: str
    downbeats: list[float]
    beats: list[float]
    bpm: float
    length_sec:float

    @classmethod
    def from_sqlite_row(cls, row):
        downbeats = json.loads(row['downbeats'])
        beats = json.loads(row['beats'])
        return cls(row['id'], row['audio_filepath'], downbeats, beats, row['bpm'], row['length'])

@dataclass
class ResultItem:
    id: int
    filename: str
    downbeats: list[float]
    beats: list[float]
    bpm: float
    length_sec:float
    audio: torch.Tensor
    downbeat_phase: torch.Tensor
    beat_phase: torch.Tensor

    @classmethod
    def from_task(cls, task:TaskItem, audio:torch.Tensor, downbeats, beats, downbeat_phase, beat_phase):
        return cls(task.id, task.filename, downbeats, beats, task.bpm, task.length_sec, audio, downbeat_phase, beat_phase)

def precompute():
    n_fft_bins = int(1+N_FFT/2)
    stft = feat.stft.STFT(
        n_fft=N_FFT,
        hop_length=HOP,
        sr = SAMPLE_RATE,
        output_format='Magnitude'
        ).to(DEVICE)
    band_filter = create_log_filter(n_fft_bins, 81).to(DEVICE)
    return (stft, band_filter)

def producer(task:TaskItem):
    """
        - Loads audio from [first_downbeat, end] as Tensor
        - Calculates beat/downbeat phase for each FFT frame
        - Puts in Result Queue
    """
    try:
        audio, downbeats, beats = load_audio_segment(task.filename, task.downbeats, task.beats, SAMPLE_RATE)
    except:
        return None
    if len(downbeats) < 4:
        return None
    audio.share_memory_()
    num_frames = 1 + (audio.shape[0]) // HOP

    downbeat_phase = calculate_beat_phase(num_frames, downbeats, SAMPLE_RATE, HOP).share_memory_()
    beat_phase = calculate_beat_phase(num_frames, beats, SAMPLE_RATE, HOP).share_memory_()

    return ResultItem.from_task(task, audio, downbeats, beats, downbeat_phase, beat_phase)

def consumer(batch:list[ResultItem], kernels):
    """
        - Computes STFT as batch
        - Filters results into freq. bands
        - Appends results to dataset file
    """
    stft, band_filter = kernels
    audio_list = [item.audio for item in batch]
    max_length = max([audio.shape[0] for audio in audio_list])
    audio_list = [torch.nn.functional.pad(audio, (0, max_length - audio.shape[0])).unsqueeze(0) for audio in audio_list]
    audio_batch = torch.cat(audio_list, dim=0).to(DEVICE)

    audio_batch.unsqueeze(1)
    stft_batch = torchaudio.functional.amplitude_to_DB(torch.abs(stft(audio_batch)), multiplier=10., amin=0.00001, db_multiplier=1)
    band_stft_batch = apply_log_filter(stft_batch, band_filter)
    with h5py.File(OUTPUT_PATH, 'a') as h5f:
        for idx, meta in enumerate(batch):
            if f'{GROUP}/{meta.id}' in h5f:
                continue
            song_spec = band_stft_batch[idx, :, :meta.downbeat_phase.shape[-1]]
            song_spec = (song_spec - song_spec.min()) / (song_spec.max() - song_spec.min())
            clip_group = h5f.create_group(f'{GROUP}/{meta.id}')
            clip_group.create_dataset('stft', data=song_spec.cpu().numpy())
            clip_group.create_dataset('downbeat_phase', data=meta.downbeat_phase.cpu().numpy())
            clip_group.create_dataset('beat_phase', data=meta.beat_phase.cpu().numpy())
            clip_group.attrs['bpm'] = meta.bpm
            clip_group.attrs['filepath'] = meta.filename
            clip_group.attrs['id'] = meta.id
            clip_group.attrs['downbeats'] = meta.downbeats
            clip_group.attrs['beats'] = meta.beats
    print('proc done ---', band_stft_batch.shape)



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    torch.cuda.empty_cache()

    with SQLiteClient() as client:
        tasks = [TaskItem.from_sqlite_row(song) for song in client.get_rendered_songs()]
    todo = []
    if os.path.exists(OUTPUT_PATH):
        with h5py.File(OUTPUT_PATH, 'r') as h5f:
            for group in [GROUP, 'train', 'test', 'val']:
                tasks = [task for task in tasks if f'{group}/{task.id}' not in h5f]
    else:
        with h5py.File(OUTPUT_PATH, 'w') as hdf5_file:
            hdf5_file.create_group(GROUP)

    # Sort by song length to greatly amount of padding / wasted STFT computation 
    tasks.sort(key=lambda task: task.length_sec, reverse=True)
    
    print(f'{len(tasks)} Tasks')

    batch_proc = BatchProcessor(tasks, producer, consumer, NUM_BATCH, NUM_WORKERS, precompute)
    batch_proc.start()
    torch.cuda.empty_cache()


    # Randomly split results into Train-Test-Val
    with h5py.File(OUTPUT_PATH, 'r+') as file:
        for name in ['train', 'val', 'test']:
            if name not in file:
                file.create_group(name)
        data_group = file['data']
        subgroups = list(data_group.keys())
        np.random.shuffle(subgroups)
        total_groups = len(subgroups)
        train_end = int(total_groups * 0.8)
        val_end = train_end + int(total_groups * 0.18)
        train_groups = subgroups[:train_end]
        val_groups = subgroups[train_end:val_end]
        test_groups = subgroups[val_end:]

        def move_groups(source_group, dest_group_name, group_names):
            dest_group_path = f'/{dest_group_name}'
            for name in group_names:
                source_path = source_group[name].name
                dest_path = f'{dest_group_path}/{name}'
                file.move(source_path, dest_path)
        move_groups(data_group, 'train', train_groups)
        move_groups(data_group, 'val', val_groups)
        move_groups(data_group, 'test', test_groups)

        print(len(file['train'].keys()), len(file['val'].keys()), len(file['test'].keys()))
