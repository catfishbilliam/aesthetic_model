import os
import re
import shutil
import csv
from pathlib import Path
from datetime import datetime
from PIL import Image, ExifTags
from typing import Optional

import imagehash
from PIL import Image, UnidentifiedImageError

def get_image_phash(filepath: Path) -> Optional[str]:
    """
    Compute the perceptual hash (pHash) of an image, if possible.
    Returns the pHash as a string, or None if it fails.
    """
    try:
        with Image.open(filepath) as img:
            return str(imagehash.phash(img))
    except (UnidentifiedImageError, OSError):
        return None

# === CONFIGURATION ===
# This script reorganizes files already in the destination folder.
DEST_DIR = "/Users/wconway/Documents/cumulus"
LOG_FILE = "destination_organize_log.csv"

# ---------------------------
# Functions for EXIF and Creation Date
# ---------------------------
def get_exif_creation_date(image_path: Path) -> Optional[datetime]:
    """
    Attempt to read the EXIF DateTimeOriginal tag from an image.
    Returns a datetime object if found, otherwise None.
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                exif = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_data.items()}
                date_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
                if date_str:
                    return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
    except Exception as e:
        print(f"  [Warning] Could not read EXIF from {image_path}: {e}")
    return None

def get_creation_date(file_path: Path) -> datetime:
    """
    Returns the creation date for an image file.
    Tries EXIF DateTimeOriginal first; if not available, falls back to file system creation time.
    """
    exif_date = get_exif_creation_date(file_path)
    if exif_date:
        return exif_date
    return datetime.fromtimestamp(file_path.stat().st_ctime)

# ---------------------------
# Grouping Key Extraction
# ---------------------------
def get_group_folder(filename: str) -> str:
    """
    Determine a grouping key based on the filename.
    
    Strategy:
    1. If the filename (without extension) matches the pattern:
         YYYYMMDD_EventName_Sequence
       then return: YYYY-MM-DD_EventName.
    2. Else, if the filename contains delimiters and ends with a numeric token,
       drop that token and join the rest.
    3. Otherwise, return "Miscellaneous".
    """
    stem = Path(filename).stem

    # Pattern 1: Strict pattern with date and event name
    pattern1 = re.compile(r'^(\d{8})_([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)_(\d+)$')
    match = pattern1.match(stem)
    if match:
        date_str = match.group(1)
        event_name = match.group(2)
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return f"{formatted_date}_{event_name}"

    # Pattern 2: Split on delimiters and drop trailing numeric token if present
    parts = re.split(r'[_\-\s]+', stem)
    if len(parts) > 1:
        if parts[-1].isdigit():
            group_key = "_".join(parts[:-1])
        else:
            group_key = "_".join(parts)
        if group_key.strip():
            return group_key

    return "Miscellaneous"

# ---------------------------
# Logging Function
# ---------------------------
def log_action(filename: str, original_path: str, new_path: str, created_at: datetime, action: str):
    """Append an action record to the CSV log."""
    log_rows.append((filename, original_path, new_path, created_at.strftime("%Y-%m-%d %H:%M:%S"), action))

# CSV log header
log_rows = [("filename", "original_path", "new_path", "created_at", "action")]

# ---------------------------
# Main Organization Function
# ---------------------------
def main():
    dest = Path(DEST_DIR)
    # Gather all files in the top level of DEST_DIR
    # (If files exist in subfolders already and you want to process them recursively, you can use rglob instead)
    all_files = [f for f in dest.iterdir() if f.is_file()]
    print(f"Found {len(all_files)} file(s) in destination folder to reorganize.")

    for file in all_files:
        # Get grouping key from the filename
        group_key = get_group_folder(file.name)
        # Get the creation date of the file
        creation_date = get_creation_date(file)
        # Use creation date (formatted as YYYY-MM) as part of the new folder name
        date_str = creation_date.strftime("%Y-%m")
        # If the group key already appears to start with a date, assume it has a date
        if group_key and group_key[0].isdigit():
            new_folder_name = group_key
        else:
            new_folder_name = f"{group_key}_{date_str}"
        # Build the new destination folder path
        target_folder = dest / new_folder_name
        target_folder.mkdir(exist_ok=True)
        # Build the new file path
        new_path = target_folder / file.name
        counter = 1
        while new_path.exists():
            stem, ext = os.path.splitext(file.name)
            new_path = target_folder / f"{stem}_{counter}{ext}"
            counter += 1

        # Move the file to the new location
        print(f"Moving '{file.name}' to folder '{target_folder.name}'")
        shutil.move(str(file), str(new_path))
        log_action(file.name, str(file), str(new_path), creation_date, "moved")

    # Write the CSV log
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.writer(f)
        writer.writerows(log_rows)
    print(f"\nOrganization complete. Log saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
import os
import re
import shutil
import csv
from pathlib import Path
from datetime import datetime
from PIL import Image, ExifTags
from typing import Optional
import imagehash

# === CONFIGURATION ===
DEST_DIR = "/Users/wconway/Documents/cumulus"
LOG_FILE = "destination_organize_log.csv"

# Allowed image file extensions (in lowercase)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp"}

# pHash similarity threshold (if difference > threshold, consider not matching)
PHASH_SIMILARITY_THRESHOLD = 10

# ---------------------------
# EXIF and Creation Date Functions
# ---------------------------
def get_exif_creation_date(image_path: Path) -> Optional[datetime]:
    """
    Attempt to read the EXIF DateTimeOriginal tag from an image.
    Returns a datetime object if found, otherwise None.
    """
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                exif = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_data.items()}
                date_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
                if date_str:
                    return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
    except Exception as e:
        print(f"  [Warning] Could not read EXIF from {image_path}: {e}")
    return None

def get_creation_date(file_path: Path) -> datetime:
    """
    Returns the creation date for an image file.
    Tries EXIF DateTimeOriginal first; if not available, falls back to the file system creation time.
    """
    exif_date = get_exif_creation_date(file_path)
    if exif_date:
        return exif_date
    return datetime.fromtimestamp(file_path.stat().st_ctime)

# ---------------------------
# Grouping Key Extraction
# ---------------------------
def get_group_folder(filename: str) -> str:
    """
    Determine a grouping key based on the filename.
    
    Strategy:
    1. If the filename (without extension) matches the pattern:
         YYYYMMDD_EventName_Sequence
       then return: YYYY-MM-DD_EventName.
    2. Else, if the filename contains common delimiters and ends with a numeric token,
       drop that token and join the rest.
    3. Otherwise, return "Miscellaneous".
    """
    stem = Path(filename).stem
    pattern1 = re.compile(r'^(\d{8})_([A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)_(\d+)$')
    match = pattern1.match(stem)
    if match:
        date_str = match.group(1)
        event_name = match.group(2)
        formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        return f"{formatted_date}_{event_name}"
    parts = re.split(r'[_\-\s]+', stem)
    if len(parts) > 1:
        if parts[-1].isdigit():
            group_key = "_".join(parts[:-1])
        else:
            group_key = "_".join(parts)
        if group_key.strip():
            return group_key
    return "Miscellaneous"

# ---------------------------
# File Identification
# ---------------------------
def is_allowed_file(filepath: Path) -> bool:
    """
    Determine if a file is a photo.
    Checks if the file name ends with an allowed extension,
    optionally followed by extra digits (e.g., '.jpg123').
    """
    lower_name = filepath.name.lower()
    for ext in ALLOWED_EXTENSIONS:
        pattern = re.compile(rf"\.{ext[1:]}(\d+)?$")
        if pattern.search(lower_name):
            return True
    return False

# ---------------------------
# Logging Function
# ---------------------------
def log_action(filename: str, original_path: str, new_path: str, created_at: datetime, action: str):
    """Append an action record to the CSV log."""
    log_rows.append((filename, original_path, new_path, created_at.strftime("%Y-%m-%d %H:%M:%S"), action))

log_rows = [("filename", "original_path", "new_path", "created_at", "action")]

# ---------------------------
# Content Similarity Check
# ---------------------------
def folder_similarity(target_folder: Path, file: Path, threshold: int = PHASH_SIMILARITY_THRESHOLD) -> bool:
    """
    Check if the file's content is similar to at least one file already in target_folder.
    Returns True if at least one file in the folder has a pHash difference <= threshold.
    If target_folder is empty, returns True.
    """
    phash_file = get_image_phash(file)
    if phash_file is None:
        return True
    for f in target_folder.iterdir():
        if f.is_file() and is_allowed_file(f):
            phash_existing = get_image_phash(f)
            if phash_existing is not None:
                diff = imagehash.hex_to_hash(phash_file) - imagehash.hex_to_hash(phash_existing)
                if diff <= threshold:
                    return True
    return False

# ---------------------------
# Main Organization Function
# ---------------------------
def main():
    dest = Path(DEST_DIR)
    # Gather all files in the top level of DEST_DIR that are photos
    all_files = [f for f in dest.iterdir() if f.is_file() and is_allowed_file(f)]
    print(f"Found {len(all_files)} file(s) in destination folder to reorganize.")

    for file in all_files:
        # Get grouping key from filename
        group_key = get_group_folder(file.name)
        # Get creation date
        creation_date = get_creation_date(file)
        date_str = creation_date.strftime("%Y-%m-%d")
        # Build initial new folder name
        if group_key and group_key[0].isdigit():
            base_folder_name = group_key
        else:
            base_folder_name = f"{group_key}_{date_str}"
        target_folder = dest / base_folder_name

        # If target folder exists and is not empty, check content similarity.
        # If file content is not similar to existing content, assign it to "Miscellaneous".
        if target_folder.exists() and any(target_folder.iterdir()):
            if not folder_similarity(target_folder, file):
                print(f"  Content mismatch for '{file.name}'; reassigning to 'Miscellaneous'")
                target_folder = dest / "Miscellaneous"
        
        target_folder.mkdir(exist_ok=True)
        # Determine destination filename (handle collisions)
        new_path = target_folder / file.name
        counter = 1
        while new_path.exists():
            stem, ext = os.path.splitext(file.name)
            new_path = target_folder / f"{stem}_{counter}{ext}"
            counter += 1
        print(f"Moving '{file.name}' to folder '{target_folder.name}'")
        shutil.move(str(file), str(new_path))
        log_action(file.name, str(file), str(new_path), creation_date, "moved")

    # Write CSV log
    with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)
    print(f"\nOrganization complete. Log saved to: {LOG_FILE}")

if __name__ == "__main__":
    main()
