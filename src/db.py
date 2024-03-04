import sqlite3
from typing import List, Optional, Dict
import json

class SQLiteClient:
    def __init__(self, db_path: str = 'data/song.db'):
        self.db_path = db_path
        self.conn = None  # Connection will be established in __enter__

    def __enter__(self):
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        # Enable Write-Ahead Logging mode for better concurrency
        self.conn.execute('PRAGMA journal_mode=WAL;')
        self.create_tables()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
        # Optionally, handle exceptions here


    def create_tables(self):
        with self.conn:  # Use the connection's context manager to automatically commit/rollback
            cursor = self.conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Song (
                    id INTEGER PRIMARY KEY,
                    midi_filepath TEXT UNIQUE NOT NULL,
                    audio_filepath TEXT,
                    downbeats TEXT,
                    beats TEXT,
                    bpm FLOAT,
                    instruments TEXT,
                    length FLOAT DEFAULT NULL,
                    has_split BOOLEAN DEFAULT FALSE,
                    has_error BOOLEAN DEFAULT FALSE
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Kit (
                    id INTEGER PRIMARY KEY,
                    sf_path TEXT NOT NULL,
                    num_presets DEFAULT 0
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Instrument (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    bank INTEGER,
                    preset INTEGER,
                    sf_path TEXT NOT NULL,
                    gm_class INTEGER DEFAULT -1,
                    kit_id INTEGER DEFAULT NULL,
                    is_drum INTEGER DEFAULT 0,
                    FOREIGN KEY (kit_id) REFERENCES Kit (id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS SongStem (
                    id INTEGER PRIMARY KEY,
                    song_id INTEGER,
                    inst_id INTEGER,
                    name TEXT,
                    program INTEGER,
                    is_drum INTEGER,
                    midi_filepath TEXT NOT NULL,
                    audio_filepath TEXT,
                    is_noise BOOLEAN DEFAULT FALSE,
                    is_silent BOOLEAN DEFAULT FALSE,
                    had_warning BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (song_id) REFERENCES Song (id),
                    FOREIGN KEY (inst_id) REFERENCES Instrument (id)
                )
            ''')
            self.conn.commit()

    def create_indices(self):
        with self.conn:  # Use the connection's context manager to automatically commit/rollback
            cursor = self.conn.cursor()
            cursor.execute('CREATE INDEX idx_songstem_song_id ON SongStem(song_id);')
            cursor.execute('CREATE INDEX idx_songstem_audio_filepath ON SongStem(audio_filepath);')
            self.conn.commit()


    def set_has_error(self, song_id):
        with self.conn:  # Use the connection's context manager to automatically commit/rollback
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE Song
                SET has_error = TRUE
                WHERE id = ?
            ''', (song_id,))
            self.conn.commit()

    def drop_song_stems_table(self):
        with self.conn:  # Use the connection's context manager to automatically commit/rollback
            cursor = self.conn.cursor()
            cursor.execute('DROP TABLE IF EXISTS SongStem')
            self.conn.commit()

    def does_path_exist(self, midi_filepath: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM Song WHERE midi_filepath = ?', (midi_filepath,))
        return cursor.fetchone()[0] > 0

    def insert_instrument(self, name: str, bank: int, preset: int, sf_path: str, kit_id:int, gm_class: int = -1):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO Instrument (name, bank, preset, sf_path, kit_id, gm_class)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, bank, preset, sf_path, kit_id, gm_class))
            self.conn.commit()
            return cursor.lastrowid

    def instrument_exists(self, name: str, bank: int, preset: int, sf_path: str) -> bool:
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT COUNT(*) FROM Instrument
            WHERE name = ? AND bank = ? AND preset = ? AND sf_path = ?
        ''', (name, bank, preset, sf_path))
        return cursor.fetchone()[0] > 0

    def update_stem_program(self, id, program):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE SongStem
                SET program = ?
                WHERE id = ?
            ''', (program, id))
            self.conn.commit()

    def insert_song(self, midi_filepath: str, audio_filepath: Optional[str], downbeats: List[float], beats: List[float], bpm: float, instruments: List[str]):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO Song (midi_filepath, audio_filepath, downbeats, beats, bpm, instruments)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (midi_filepath, audio_filepath, json.dumps(list(downbeats)), json.dumps(list(beats)), bpm, json.dumps(instruments)))
            self.conn.commit()
            return cursor.lastrowid

    def insert_song_stem(self, song_id: int, midi_filepath: str, name:str, program:int, is_drum:int):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT INTO SongStem (song_id, midi_filepath, name, program, is_drum)
                VALUES (?, ?, ?, ?, ?)
            ''', (song_id, midi_filepath, name, program, is_drum))
            self.conn.commit()
            return cursor.lastrowid

    def insert_song_stems(self, song_id, song_stems: List[Dict]):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.executemany('''
                INSERT INTO SongStem (song_id, midi_filepath, name, program, is_drum)
                VALUES (:song_id, :midi_filepath, :name, :program, :is_drum)
            ''', song_stems)
            cursor.execute('''
                UPDATE Song
                SET has_split = True
                WHERE id = ?
            ''', (song_id,))
            self.conn.commit()

    def get_song_by_id(self, song_id: int):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM Song WHERE id = ?', (song_id,))
        return cursor.fetchone()

    def get_song_stem_by_id(self, stem_id: int):
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM SongStem WHERE id = ?', (stem_id,))
        return cursor.fetchone()

    def get_song_with_stem_ids(self, song_id: int):
        song = self.get_song_by_id(song_id)
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM SongStem WHERE song_id = ?', (song_id,))
        stems = cursor.fetchall()
        return song, [stem['id'] for stem in stems]

    def get_stem_ids(self, song_id: int):
        cursor = self.conn.cursor()
        cursor.execute('SELECT id FROM SongStem WHERE song_id = ?', (song_id,))
        stems = cursor.fetchall()
        return [stem['id'] for stem in stems]

    def update_audio_filepath(self, id: int, audio_filepath: str, is_noise=False, is_silent=False, had_warning=False):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute(f'''
                UPDATE SongStem
                SET audio_filepath = ?, is_noise = ?, is_silent = ?, had_warning = ?
                WHERE id = ?
            ''', (audio_filepath, is_noise, is_silent, had_warning, id))
            self.conn.commit()

    def update_song_audio_filepath(self, id: int, audio_filepath: str):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute(f'''
                UPDATE Song
                SET audio_filepath = ?
                WHERE id = ?
            ''', (audio_filepath, id))
            self.conn.commit()

    def update_midi_filepath(self, id: int, midi_filepath: str):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute(f'''
                UPDATE Song
                SET midi_filepath = ?
                WHERE id = ?
            ''', (midi_filepath, id))
            self.conn.commit()

    def update_inst_gm(self, id: int, gm_class: int):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute(f'''
                UPDATE Instrument
                SET gm_class = ?
                WHERE id = ?
            ''', (gm_class, id))
            self.conn.commit()

    def update_inst_is_drum(self, id: int, is_drum: int):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute(f'''
                UPDATE Instrument
                SET is_drum = ?
                WHERE id = ?
            ''', ((1 if is_drum else 0), id))
            self.conn.commit()

    def update_song_length(self, id: int, length: float):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute(f'''
                UPDATE Song
                SET length = ?
                WHERE id = ?
            ''', (length, id))
            self.conn.commit()

    def get_all_songs_w_drums(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM Song WHERE instruments LIKE '%\"is_drum\": true%' AND midi_filepath NOT LIKE '%superslakh%'")
        return cursor.fetchall()

    def get_rendered_songs(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM Song WHERE audio_filepath IS NOT NULL")
        return cursor.fetchall()

    def get_all_songs(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM Song")
        return cursor.fetchall()

    def get_unrendered_song_ids(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM Song WHERE audio_filepath is NULL")
        return cursor.fetchall()

    def get_songs_to_split(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, midi_filepath FROM Song WHERE has_split = 0 AND has_error = 0")
        return cursor.fetchall()

    def drop_songs_by_length(self):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute("DELETE FROM Song WHERE length IS NULL")
            cursor.execute("DELETE FROM Song WHERE length <30")
            cursor.execute("DELETE FROM Song WHERE length >360")
            cursor.execute("DELETE FROM SongStem WHERE song_id NOT IN (SELECT id FROM Song)")
            self.conn.commit()

    def get_all_stems(self, cols=None):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(f'SELECT * FROM SongStem')
            return cursor.fetchall()

    def get_unassigned_stems(self, cols=None):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT * FROM SongStem WHERE inst_id IS NULL")
            # cursor.execute(f"SELECT * FROM SongStem WHERE audio_filepath IS NULL OR TRIM(audio_filepath) = ''")
            return cursor.fetchall()


    def get_all_kits(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(f'SELECT * FROM Kit')
            return cursor.fetchall()

    def get_all_insts_for_kit(self, kit_id):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(f'SELECT * FROM Instrument WHERE kit_id = ?', (kit_id,))
            return cursor.fetchall()

    def get_all_instruments(self):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(f'SELECT * FROM Instrument')
            return cursor.fetchall()

    def set_has_split(self, has_split):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute(f'''
                UPDATE Song
                SET has_split = ?
            ''', (1 if has_split else 0,))
            self.conn.commit()

    def unset_inst_ids(self):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute(f'''
                UPDATE SongStem
                SET inst_id = NULL, audio_filepath = NULL
            ''')
            self.conn.commit()

    def unset_audio_paths(self):
        with self.conn:  # Automatic commit/rollback
            cursor = self.conn.cursor()
            cursor.execute(f'''
                UPDATE Song
                SET audio_filepath = NULL
            ''')
            self.conn.commit()

    def update_stem_inst_ids(self, rows):
        with self.conn:
            cursor = self.conn.cursor()
            batch_size = 512
            for i in range(0, len(rows), batch_size):
                batch = rows[i:i+batch_size]
                cursor.executemany('''
                    UPDATE SongStem
                    SET inst_id = :inst_id
                    WHERE id = :id
                ''', batch)
            self.conn.commit()

    def get_stems_for_instrument(self, inst_id):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(f'SELECT * FROM SongStem WHERE inst_id = ?', (inst_id,))
            return cursor.fetchall()

    def get_stem_and_instrument(self, stem_id):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(f'SELECT * FROM SongStem WHERE id = ?', (stem_id,))
            stem = cursor.fetchone()
            cursor.execute(f'SELECT * FROM Instrument WHERE id = ?', (stem['inst_id'],))
            instrument = cursor.fetchone()
            return stem, instrument


    def get_stems_and_instruments(self, song_id):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(f'SELECT * FROM SongStem WHERE song_id = ?', (song_id,))
            stems = cursor.fetchall()
            stems_and_instruments = []
            for stem in stems:
                cursor.execute(f'SELECT * FROM Instrument WHERE id = ?', (stem['inst_id'],))
                instrument = cursor.fetchone()
                stems_and_instruments.append((stem, instrument))
            return stems_and_instruments

    def get_songs_with_all_non_null_stems(self):
        """
        Queries the database and retrieves a list of all Songs
        for which every related SongStem has a non-None 'audio_filepath'.

        :param connection: SQLite database connection object
        :return: List of qualifying songs
        """
        songs_query = """
        SELECT *
        FROM Song
        WHERE NOT EXISTS (
            SELECT 1
            FROM SongStem
            WHERE SongStem.song_id = Song.id
            AND (SongStem.audio_filepath IS NULL OR TRIM(SongStem.audio_filepath) = '')
        )
        """

        cursor = self.conn.cursor()
        cursor.execute(songs_query)
        songs = cursor.fetchall()  # Returns a list of tuples representing the songs
        cursor.close()
        return songs


    def get_song_ids_with_null_stems(self):
        songs_query = """
        SELECT id
        FROM Song
        WHERE EXISTS (
            SELECT 1
            FROM SongStem
            WHERE SongStem.song_id = Song.id
            AND (SongStem.audio_filepath IS NULL OR TRIM(SongStem.audio_filepath) = '')
        )
        """
        cursor = self.conn.cursor()
        cursor.execute(songs_query)
        songs = cursor.fetchall()  # Returns a list of tuples representing the songs
        cursor.close()
        return [s['id'] for s in songs]

    def insert_kit(self, sf_path):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('INSERT INTO Kit (sf_path) VALUES (?)', (sf_path,))
            kit_id = cursor.lastrowid
            cursor.execute('UPDATE Instrument SET kit_id = ? WHERE sf_path = ?', (kit_id, sf_path))
            self.conn.commit()
            cursor.close()
            return kit_id

    def kits_per_song(self):
        query = """
        SELECT
            Song.id,
            COUNT(DISTINCT Instrument.kit_id) AS num_kits
        FROM
            Song
        JOIN
            SongStem ON Song.id = SongStem.song_id
        JOIN
            Instrument ON SongStem.inst_id = Instrument.id
        WHERE
            Instrument.kit_id IS NOT NULL
        GROUP BY
            Song.id
        """
        cursor = self.conn.cursor()
        cursor.execute(query)
        songs = cursor.fetchall()  # Returns a list of tuples representing the songs
        cursor.close()
        return songs