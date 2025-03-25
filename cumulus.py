import os
import zipfile
import hashlib
import shutil
import tempfile
import csv
from datetime import datetime
from pathlib import Path
from PIL import Image, UnidentifiedImageError
import imagehash

# === CONFIGURATION ===
SOURCE_DIR = "/Users/wconway/Library/CloudStorage/Box-Box/box-group-engr-dean-communications/Cumulus assets/Clark School"
DEST_DIR = "/Users/wconway/Documents/cumulus"
LOG_FILE = "asset_log.csv"

# Allowed image file extensions (in lowercase)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp"}

# Dictionaries to track seen files/hashes
seen_hashes = {}   # For exact duplicate detection
seen_phashes = {}  # For perceptual duplicate detection

# CSV log header: filename, original_path, status, sha256, phash, copied_to, created_at
log_rows = [("filename", "original_path", "status", "sha256", "phash", "copied_to", "created_at")]



def get_file_hash(filepath):
    """Compute SHA-256 hash of a file."""
    hash_func = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()


def get_image_phash(filepath):
    """Compute perceptual hash (pHash) of an image, if possible."""
    try:
        with Image.open(filepath) as img:
            return str(imagehash.phash(img))
    except (UnidentifiedImageError, OSError):
        return None


def get_creation_date(filepath):
    """Get the file creation date as a datetime object."""
    timestamp = os.path.getctime(filepath)
    return datetime.fromtimestamp(timestamp)


def log_asset(filename, original_path, status, sha256, phash, copied_to, created_at):
    """Append asset information to the CSV log."""
    log_rows.append((filename, original_path, status, sha256, phash or "N/A", copied_to or "N/A", created_at))


def is_allowed_file(filepath):
    """Return True if the file extension is in the allowed image formats."""
    ext = Path(filepath).suffix.lower()
    return ext in ALLOWED_EXTENSIONS


def organize_file(filepath):
    """
    Process a single file if it is a photo:
      - Compute SHA-256 and perceptual hash
      - Check for exact or near duplicates
      - Copy the file into DEST_DIR under a subfolder named by creation date
      - Log the action.
    Returns True if the file was successfully processed.
    """
    if not is_allowed_file(filepath):
        print(f"Skipping non-photo file: {filepath}")
        return False

    print(f"Processing photo: {filepath}")
    sha256 = get_file_hash(filepath)
    phash = get_image_phash(filepath)
    filename = os.path.basename(filepath)
    creation_date = get_creation_date(filepath)
    subfolder_name = creation_date.strftime("%Y-%m")
    dest_folder = Path(DEST_DIR) / subfolder_name
    dest_folder.mkdir(parents=True, exist_ok=True)
    dest_path = dest_folder / filename

    # Skip if exact duplicate detected
    if sha256 in seen_hashes:
        print(f"  [Exact Duplicate] Skipping: {filepath}")
        log_asset(filename, filepath, "duplicate_exact", sha256, phash, None, creation_date)
        return False

    # Near-duplicate detection using perceptual hash
    near_dup = False
    if phash:
        for existing_phash in seen_phashes:
            distance = imagehash.hex_to_hash(phash) - imagehash.hex_to_hash(existing_phash)
            if distance <= 5:  # Hamming distance threshold (0-5 considered similar)
                near_dup = True
                print(f"  [Near Duplicate] Detected (distance {distance}): {filepath}")
                break

    # Handle filename collisions
    counter = 1
    while dest_path.exists():
        stem, ext = os.path.splitext(filename)
        dest_path = dest_folder / f"{stem}_{counter}{ext}"
        counter += 1

    shutil.copy2(filepath, dest_path)
    print(f"  [Copied] {filepath} → {dest_path}")

    seen_hashes[sha256] = filepath
    if phash:
        seen_phashes[phash] = filepath

    status = "copied_near_duplicate" if near_dup else "copied"
    log_asset(filename, filepath, status, sha256, phash, str(dest_path), creation_date)
    return True


def organize_file_with_override(data_filepath, folder_name):
    """
    Process an asset stored as a folder.
    The folder name (e.g., "20131106_Hackathon_291.JPG") is used as the original photo name.
    The candidate file (typically "00000001.data") is converted to PNG.
    """
    print(f"\n[Folder-Override] Processing asset directory: '{folder_name}' using candidate: {data_filepath}")
    # Derive the proper file name by keeping the original (folder) name but converting to PNG.
    stem, _ = os.path.splitext(folder_name)
    final_filename = f"{stem}.png"  # Force .png extension
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, final_filename)

    # Open the candidate file with Pillow and save it as PNG.
    try:
        with Image.open(data_filepath) as img:
            # Optionally, convert to a standard mode (e.g., "RGB") if needed:
            if img.mode in ("RGBA", "P"):
                img = img.convert("RGB")
            img.save(temp_path, "PNG")
            print(f"  Converted image to PNG and saved to temp path: {temp_path}")
    except Exception as e:
        print(f"  [Error] Failed to convert {data_filepath} to PNG: {e}")
        return False

    # Process the temporary PNG file
    success = organize_file(temp_path)

    # Clean up temporary file
    try:
        os.remove(temp_path)
        print(f"  Removed temp file: {temp_path}")
    except OSError as e:
        print(f"  [Warning] Could not remove temp file: {e}")

    return success


def extract_and_process_zip(zip_path):
    """
    Extract a ZIP file and process its contents recursively.
    Returns the number of photo assets processed from the ZIP.
    """
    print(f"\nExtracting ZIP: {zip_path}")
    saved_count = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            print(f"  Extracted to temporary folder: {temp_dir}")
            for root, dirs, files in os.walk(temp_dir):
                # Process asset directories (folders named with an allowed extension)
                asset_dirs = [d for d in dirs if Path(d).suffix.lower() in ALLOWED_EXTENSIONS]
                for d in asset_dirs:
                    asset_dir_path = os.path.join(root, d)
                    print(f"\n[ZIP] Found asset directory: {asset_dir_path}")
                    candidate = None
                    # Look for a candidate file (e.g., ending in .data)
                    for f in os.listdir(asset_dir_path):
                        if f.lower().endswith(".data"):
                            candidate = os.path.join(asset_dir_path, f)
                            break
                    if candidate:
                        print(f"  Found candidate: {candidate}")
                        if organize_file_with_override(candidate, d):
                            saved_count += 1
                    else:
                        print(f"  No candidate (.data) found in {asset_dir_path}; skipping.")
                    # Prevent further recursion into this asset directory
                    dirs.remove(d)
                # Process remaining normal files
                for file in files:
                    full_path = os.path.join(root, file)
                    if organize_file(full_path):
                        saved_count += 1
        except zipfile.BadZipFile:
            print(f"  [Error] Bad ZIP file, skipping: {zip_path}")
    print(f"→ {saved_count} photo(s) saved from ZIP file: {zip_path}\n")
    return saved_count


def main():
    print(f"Starting scan of archive: {SOURCE_DIR}\n")
    total_zips = 0
    # Walk through the entire directory tree under SOURCE_DIR
    for root, dirs, files in os.walk(SOURCE_DIR):
        print(f"Scanning folder: {root}")
        # Process asset directories first (folders named with an allowed image extension)
        asset_dirs = [d for d in dirs if Path(d).suffix.lower() in ALLOWED_EXTENSIONS]
        for d in asset_dirs:
            asset_dir_path = os.path.join(root, d)
            candidate = None
            for f in os.listdir(asset_dir_path):
                if f.lower().endswith(".data"):
                    candidate = os.path.join(asset_dir_path, f)
                    break
            if candidate:
                print(f"\nProcessing asset directory: {asset_dir_path} using candidate: {candidate}")
                organize_file_with_override(candidate, d)
            else:
                print(f"Asset directory {asset_dir_path} has no .data file; skipping.")
            # Remove from recursion
            dirs.remove(d)
        # Process remaining files in the current folder
        for file in files:
            full_path = os.path.join(root, file)
            if file.lower().endswith(".zip"):
                total_zips += 1
                print(f"\nFound ZIP file #{total_zips}: {full_path}")
                extract_and_process_zip(full_path)
            else:
                organize_file(full_path)
    print(f"\nScan complete. Total ZIP files processed: {total_zips}")

    # Write the CSV log
    with open(LOG_FILE, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(log_rows)
    print(f"CSV log saved to: {LOG_FILE}")


if __name__ == "__main__":
    main()
