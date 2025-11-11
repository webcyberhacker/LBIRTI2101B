"""Download functionality for Xeno-Canto recordings."""

import requests
from pathlib import Path
from .config import RECORDINGS_DIR
from .utils import normalize_url, sanitize_filename


def download_recording(recording, output_dir):
    """Download a single recording and save it to the output directory."""
    recording_id = recording.get('id')
    file_url = recording.get('file', '')
    
    if not file_url:
        return False
    
    file_url = normalize_url(file_url)
    
    # Get or create filename
    original_filename = recording.get('file-name', '')
    safe_filename = sanitize_filename(original_filename) if original_filename else None
    
    if not safe_filename:
        safe_filename = f"XC{recording_id}.mp3"
    elif not safe_filename.lower().endswith('.mp3'):
        safe_filename = f"{safe_filename}.mp3"
    
    file_path = Path(output_dir) / safe_filename
    
    # Skip if file already exists
    if file_path.exists():
        return True
    
    try:
        response = requests.get(file_url, stream=True, timeout=30)
        response.raise_for_status()
        
        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return True
    except Exception:
        # Clean up partial file if it exists
        if file_path.exists():
            file_path.unlink()
        return False


def download_recordings(recordings, bird_name, output_dir=None):
    """
    Download all recordings and save them to recordings/bird_name/.
    
    Args:
        recordings: List of recording dictionaries
        bird_name: Name for the subdirectory
        output_dir: Base output directory (default: RECORDINGS_DIR)
    
    Returns:
        tuple: (downloaded_count, failed_count)
    """
    if output_dir is None:
        output_dir = Path(RECORDINGS_DIR) / bird_name
    else:
        output_dir = Path(output_dir) / bird_name
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    downloaded_count = 0
    failed_count = 0
    
    for recording in recordings:
        if download_recording(recording, output_dir):
            downloaded_count += 1
        else:
            failed_count += 1
    
    return downloaded_count, failed_count

