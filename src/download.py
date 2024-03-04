import requests
from pathlib import Path
from bs4 import BeautifulSoup
import os
import tarfile
from multiprocessing import Pool

def download_file(url):
    filename = url.split("/")[-1]
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded {filename}")

def extract_tar_gz(file_path):
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall()
        print(f"Extracted {file_path}")

def get_sf2_links(url):
    try:
        response = requests.get(url)
        response.raise_for_status() 
        soup = BeautifulSoup(response.text, 'html.parser')
        links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.sf2')]
        return links
    except requests.RequestException as e:
        return f"Error fetching page: {e}"

def download_sf(filename, base_url, target_dir):
    try:
        file_url = f"{base_url.rstrip('/')}/{filename}"
        response = requests.get(file_url, stream=True)
        response.raise_for_status()
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        file_path = os.path.join(target_dir, filename.split('/')[-1])  # Get the last part if filename is a path
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} to {file_path}")
    except requests.RequestException as e:
        print(f"Error downloading {filename}: {e}")

def download_sf2_files(base_url, filenames, target_dir):
    with Pool(processes=8) as p:
        p.starmap(download_sf, [(filename, base_url, target_dir) for filename in filenames])


def download_midi_and_soundfonts():
    midi_url = "http://hog.ee.columbia.edu/craffel/lmd/lmd_full.tar.gz"
    download_file(midi_url)
    file_path = "lmd_full.tar.gz"
    extract_tar_gz(file_path)
    os.rename('lmd_full', 'midi')
    # Delete the tar file after extraction
    os.remove(file_path)

    sf_url = "https://archive.org/download/free-soundfonts-sf2-2019-04"
    sf2_links = get_sf2_links(sf_url)
    os.makedirs('soundfonts', exist_ok=True)
    target_dir = "soundfonts"
    download_sf2_files(sf_url, sf2_links, target_dir)