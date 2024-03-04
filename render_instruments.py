import multiprocessing
import os
import pretty_midi
import shutil
from src.audio.render import make_synth, fluidsynthesize, save_audio, mix_audios
from src.db import SQLiteClient

SAMPLE_RATE = 22050

def render_stem(stem, synthesizer, sfid, tracknum):
    output_path = stem['midi_filepath'].replace('/midi/', '/audio/').replace('.mid', '.flac')
    if os.path.isfile(output_path):
        print('stem -- '+stem['name'])
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    inst = pretty_midi.PrettyMIDI(stem['midi_filepath']).instruments[0]

    audio = fluidsynthesize(inst, fs=SAMPLE_RATE, synthesizer=synthesizer, sfid=sfid, channel=tracknum)

    save_audio(audio, output_path, normalize=True, sr=SAMPLE_RATE)
    print('stem -- '+stem['name'])
    return output_path

def render_song(song_id):
    with SQLiteClient('data/song.db') as client:
        stems_and_instruments = client.get_stems_and_instruments(song_id)
    stems_and_instruments.sort(key=lambda elem: elem[1]['kit_id'])

    stem_paths = []

    current_sf_path = stems_and_instruments[0][1]['sf_path']
    synth, sfid = make_synth(current_sf_path, sr=SAMPLE_RATE)
    for (stem, instrument) in stems_and_instruments:
        if stem['audio_filepath'] is not None:
            stem_paths.append(stem['audio_filepath'])
            continue
        if instrument['sf_path'] != current_sf_path:
            synth.delete()
            del synth
            current_sf_path = instrument['sf_path']
            synth, sfid = make_synth(current_sf_path)
        tracknum = 9 if stem['is_drum'] else 0
        synth.program_select(tracknum, sfid, instrument['bank'], instrument['preset'])
        output_path = render_stem(stem, synth, sfid, tracknum)
        stem_paths.append(output_path)

    output_path = os.path.join(stem_paths[0].split('instruments')[0], 'full.flac')
    mix_audios(stem_paths, output_path)
    print('mixed', output_path)

    shutil.rmtree(os.path.join(os.path.dirname(output_path), 'instruments'), ignore_errors=True)
    with SQLiteClient('data/song.db') as client:
        client.update_song_audio_filepath(song_id, output_path)

def main():
    with SQLiteClient('data/song.db') as client:
        song_ids = [song['id'] for song in client.get_unrendered_song_ids()]
    print(len(song_ids))
    with multiprocessing.Pool(8) as p:
        p.map(render_song, song_ids)

if __name__ == "__main__":
    main()
