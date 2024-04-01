from src.batch_proc import BatchProcessor
from src.audio.io import split_audio_by_tempo
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
from deeprhythm.audio_proc.hcqm import compute_hcqm, make_kernels
from deeprhythm.utils import load_and_split_audio
NUM_WORKERS = 14
NUM_BATCH = 64
DEVICE = 'cuda'
SAMPLE_RATE = 22050
N_FFT = 2048
HOP = 512
OUTPUT_PATH = 'hcqm_data_big.hdf5'
GROUP = 'data'

@dataclass
class TaskItem:
    id: int
    audio_file: str
    midi_file:str
    bpm: float
    length_sec:float

    @classmethod
    def from_sqlite_row(cls, row):
        return cls(row['id'], row['audio_filepath'], row['midi_filepath'], row['bpm'], row['length'])

@dataclass
class ResultItem:
    id: int
    filename: str
    clips: torch.Tensor
    bpm: float
    num_clips:float

    @classmethod
    def from_task(cls, task:TaskItem, clips:torch.Tensor, num_clips:int, bpm:int):
        return cls(f'{task.id}-{bpm}', task.audio_file, clips, bpm, num_clips)

def precompute():
    stft, band_filter, cqt = make_kernels()
    return (stft, band_filter, cqt)

def producer(task:TaskItem):
    """
        - Loads audio from [first_downbeat, end] as Tensor
        - 
        - Puts in Result Queue
    """
    try:
        results = []
        splits = split_audio_by_tempo(task.audio_file, task.midi_file)
        clip_samples = SAMPLE_RATE * 8
        for bpm, audio in splits:
            clips = []
            for i in range(0, len(audio), clip_samples):
                if i + clip_samples <= len(audio):
                    clip_tensor = torch.tensor(audio[i:i + clip_samples], dtype=torch.float32)
                    clips.append(clip_tensor)
            if clips:
                stacked_clips = torch.stack(clips, dim=0)
                stacked_clips.share_memory_()                
                results.append(ResultItem.from_task(task, stacked_clips, stacked_clips.shape[0], bpm))
    except Exception as e:
        print(e)
        return None    
    return results

def consumer(batch:list[ResultItem], kernels):
    """
        - Computes STFT as batch
        - Filters results into freq. bands
        - Appends results to dataset file
    """
    stft, band_filter, cqt = kernels
    audio_list = [item.clips for item in batch]
    audio_batch = torch.cat(audio_list, dim=0).to(DEVICE)

    hcqm = compute_hcqm(audio_batch, stft, band_filter, cqt)
    with h5py.File(OUTPUT_PATH, 'a') as h5f:
        curr_idx = 0
        for idx, meta in enumerate(batch):
            if f'{GROUP}/{meta.id}' in h5f:
                continue
            song_hcqm = hcqm[curr_idx:curr_idx+meta.num_clips, :, :, :]
            curr_idx += meta.num_clips
            clip_group = h5f.create_group(f'{GROUP}/{meta.id}')
            clip_group.create_dataset('hcqm', data=song_hcqm.cpu().numpy())
            clip_group.attrs['bpm'] = meta.bpm
            clip_group.attrs['filepath'] = meta.filename
            clip_group.attrs['id'] = meta.id
    print('proc done ---', hcqm.shape)



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
