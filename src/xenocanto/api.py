"""Xeno-Canto API v3 client."""

import requests
from .config import API_BASE_URL
from .utils import get_api_key


class XenoCantoAPI:
    """Client for Xeno-Canto API v3."""
    
    def __init__(self, api_key=None):
        """Initialize API client with optional API key."""
        self.api_key = api_key or get_api_key()
        self.base_url = API_BASE_URL
    
    def search(self, query, per_page=100, page=1):
        """
        Search for recordings.
        
        Args:
            query: Search query string (e.g., 'sp:"Turdus merula" cnt:belgium')
            per_page: Number of results per page (50-500, default 100)
            page: Page number (default 1)
        
        Returns:
            dict: API response with recordings, or None on error
        """
        params = {
            "query": query,
            "key": self.api_key,
            "per_page": per_page,
            "page": page
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return None

