"""Utility functions for Xeno-Canto operations."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_api_key():
    """Get API key from environment variable or return demo key."""
    api_key = os.getenv("XENO_CANTO_API_KEY")
    if not api_key:
        return "demo"
    return api_key


def get_bird_name(recording):
    """Extract bird name from recording for directory naming."""
    gen = recording.get('gen', '').strip()
    sp = recording.get('sp', '').strip()
    if gen and sp:
        return f"{gen}_{sp}".replace(' ', '_')
    # Fallback to English name if scientific name not available
    en_name = recording.get('en', 'unknown').strip()
    return en_name.replace(' ', '_')


def normalize_url(url):
    """Convert protocol-relative or relative URLs to absolute HTTPS URLs."""
    if url.startswith('//'):
        return f"https:{url}"
    elif url.startswith('/'):
        return f"https://xeno-canto.org{url}"
    return url


def sanitize_filename(filename):
    """Sanitize filename for filesystem compatibility."""
    safe = "".join(c for c in filename if c.isalnum() or c in "._- ")
    safe = safe.strip()
    return safe if safe and len(safe) >= 5 else None

