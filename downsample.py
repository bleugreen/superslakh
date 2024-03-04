import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

def resample_flac(file_path):
    try:
        temp_file = str(file_path) + "_temp.flac"
        # ffmpeg command to resample the .flac file
        command = ['ffmpeg', '-i', str(file_path), '-ar', '22050', '-compression_level', str(8),'-y', temp_file]
        subprocess.run(command, check=True)
        # Replace original file with the resampled file
        os.replace(temp_file, file_path)
        print(f"Resampled {file_path} to 22050Hz")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {file_path}: {e}")

def main(target_dir):
    flac_files = list(Path(target_dir).rglob('*.flac'))

    with Pool(processes=12) as pool:
        pool.map(resample_flac, flac_files)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
        main(target_dir)
    else:
        print("Usage: script.py <target_directory>")
