import os
import requests
import zipfile
import tarfile
import re
from pathlib import Path
import shutil

def download_and_extract():
    # URL of the dataset
    url = "https://kilthub.cmu.edu/ndownloader/articles/14226830/versions/2"
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download file
    print(f"Downloading data from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check if download was successful
    
    # Determine the filename from Content-Disposition header or use a default name
    if 'Content-Disposition' in response.headers:
        content_disposition = response.headers['Content-Disposition']
        filename_match = re.search(r'filename=(.+)', content_disposition)
        if filename_match:
            filename = filename_match.group(1).strip('"')
        else:
            filename = "data.zip"
    else:
        filename = "data.zip"
    
    file_path = data_dir / filename
    
    # Save the downloaded file
    with open(file_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded file saved to {file_path}")
    
    # Extract based on file extension
    print(f"Extracting {filename}...")
    if filename.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    elif filename.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(file_path, 'r:gz') as tar_ref:
            tar_ref.extractall(data_dir)
    elif filename.endswith('.tar'):
        with tarfile.open(file_path, 'r') as tar_ref:
            tar_ref.extractall(data_dir)
    else:
        print(f"Unsupported file format: {filename}. Cannot extract automatically.")
        return
    
    print(f"Files extracted to {data_dir}")
    
    # Uncomment the line below if you want to delete the archive after extraction
    # os.remove(file_path)

if __name__ == "__main__":
    download_and_extract()
    print("Data download and extraction complete!")