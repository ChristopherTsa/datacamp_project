import os
import requests
import zipfile
import re
import shutil
from pathlib import Path

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
    
    # Extract zip file directly to data directory
    print(f"Extracting {filename} directly to data folder...")
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            # Extract each file directly to the data directory
            # Skip directory entries in the zip
            if not member.endswith('/'):
                # Get just the filename without any subdirectory structure
                filename = os.path.basename(member)
                # Extract the file directly to the data directory
                source = zip_ref.open(member)
                target = open(os.path.join(data_dir, filename), "wb")
                with source, target:
                    shutil.copyfileobj(source, target)
    
    # Count the extracted files
    csv_files = list(data_dir.glob('*.csv'))
    txt_files = list(data_dir.glob('*.txt'))
    print(f"Extraction complete. Found {len(csv_files)} CSV files and {len(txt_files)} TXT files in {data_dir}")
    print(f"The original zip file has been kept at {file_path}")

if __name__ == "__main__":
    download_and_extract()
    print("Data download and extraction complete!")