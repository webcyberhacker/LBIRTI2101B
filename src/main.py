"""Main entry point for Xeno-Canto recording downloader."""

from src.xenocanto import XenoCantoAPI, download_recordings
from src.xenocanto.utils import get_bird_name


def main():
    """Main function to search and download recordings."""
    # Initialize API client
    api = XenoCantoAPI()
    
    # Search for Turdus merula in Belgium
    query = 'sp:"Parus major" cnt:belgium'
    print(f"Searching: {query}")
    
    data = api.search(query, per_page=5)
    
    if not data or not data.get('recordings'):
        print("No recordings found.")
        return
    
    recordings = data.get('recordings', [])
    total = data.get('numRecordings', 0)
    
    print(f"Found {total} recording(s). Downloading {len(recordings)}...")
    
    # Get bird name and download
    bird_name = get_bird_name(recordings[0])
    downloaded, failed = download_recordings(recordings, bird_name)
    
    print(f"\nDownloaded: {downloaded}, Failed: {failed}")


if __name__ == "__main__":
    main()

