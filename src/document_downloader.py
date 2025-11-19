import requests
import os
from pathlib import Path
from urllib.parse import urlparse, unquote
import time
from tqdm import tqdm

class DocumentDownloader:
    """Download documents from URLs with retry logic and proper naming"""
    
    def __init__(self, download_dir="downloaded_docs", max_retries=3, timeout=30):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.timeout = timeout
        self.downloaded_files = []
    
    def _get_filename_from_url(self, url):
        """Extract filename from URL"""
        parsed = urlparse(url)
        filename = unquote(os.path.basename(parsed.path))
        
        # If no filename or extension, use domain + timestamp
        if not filename or '.' not in filename:
            domain = parsed.netloc.replace('.', '_')
            timestamp = int(time.time())
            filename = f"{domain}_{timestamp}.pdf"
        
        return filename
    
    def download_file(self, url, custom_filename=None):
        """Download a single file from URL with retry logic"""
        filename = custom_filename or self._get_filename_from_url(url)
        filepath = self.download_dir / filename
        
        # Skip if already downloaded
        if filepath.exists():
            print(f"⏭️  Already exists: {filename}")
            return str(filepath)
        
        for attempt in range(self.max_retries):
            try:
                print(f"⬇️  Downloading: {url}")
                response = requests.get(url, timeout=self.timeout, stream=True)
                response.raise_for_status()
                
                # Get file size for progress bar
                total_size = int(response.headers.get('content-length', 0))
                
                # Download with progress bar
                with open(filepath, 'wb') as f:
                    if total_size > 0:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                                pbar.update(len(chunk))
                    else:
                        f.write(response.content)
                
                print(f"✅ Downloaded: {filename}")
                self.downloaded_files.append(str(filepath))
                return str(filepath)
                
            except Exception as e:
                print(f"❌ Attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"❌ Failed to download: {url}")
                    return None
    
    def download_batch(self, urls):
        """Download multiple files from a list of URLs"""
        print(f"\n📥 Downloading {len(urls)} documents...")
        downloaded_paths = []
        
        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}]")
            filepath = self.download_file(url)
            if filepath:
                downloaded_paths.append(filepath)
        
        print(f"\n✅ Successfully downloaded {len(downloaded_paths)}/{len(urls)} files")
        return downloaded_paths
    
    def get_downloaded_files(self):
        """Return list of all downloaded file paths"""
        return self.downloaded_files
