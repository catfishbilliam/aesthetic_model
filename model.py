import os
import re
import csv
import shutil
import difflib
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ExifTags, UnidentifiedImageError
import imagehash
from sklearn.cluster import KMeans

# === CONFIGURATION ===
SOURCE_DIR = "/Users/wconway/Library/CloudStorage/Box-Box/cumulus_transfer"      # Folder with downloaded photos
ORGANIZED_DIR = "/Users/wconway/Documents/cumulus_organized"                     # Folder to store organized subfolders
CURATED_DIR = "/Users/wconway/Documents/curated_photos"                          # Folder to store curated selections
ORG_LOG_FILE = "organization_log.csv"                                            # CSV log for organization
CUR_LOG_FILE = "curated_log.csv"                                                 # CSV log for curated selections

# Allowed image file extensions (in lowercase)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp"}

# pHash threshold for content-check (if difference <= threshold, images are similar)
PHASH_THRESHOLD = 10

# Similarity threshold for grouping keys (for filename-based grouping)
SIMILARITY_THRESHOLD = 0.7

# ---------------------------
# Advanced Aesthetic Model Setup
# ---------------------------
try:
    # Example: Load an advanced aesthetic predictor from Torch Hub.
    # (Replace 'smoosch/awesome-aesthetic-predictor' and 'aesthetic_predictor' with your model if needed.)
    aesthetic_model = torch.hub.load('catfishbilliam/aesthetic_model', 'aesthetic_model', pretrained=True)
    aesthetic_model.eval()
    aesthetic_model.to('cpu')
    print("Advanced aesthetic model loaded successfully.")
except Exception as e:
    print(f"[Warning] Could not load advanced aesthetic model: {e}")
    aesthetic_model = None

aesthetic_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def compute_aesthetic_score_advanced(image_path: Path) -> float:
    """
    Compute an aesthetic quality score using an advanced ML model.
    Falls back to a Laplacian variance score if the advanced model is unavailable.
    """
    if aesthetic_model is None:
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    try:
        image = Image.open(image_path).convert("RGB")
        input_tensor = aesthetic_transform(image).unsqueeze(0)
        with torch.no_grad():
            output = aesthetic_model(input_tensor.to('cpu'))
            score = output.item()  # Assumes a scalar output
            return score
    except Exception as e:
        print(f"[Error] Aesthetic scoring failed for {image_path}: {e}")
        img = cv2.imread(str(image_path))
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

# ---------------------------
# EXIF / Creation Date Functions
# ---------------------------
def get_exif_creation_date(image_path: Path) -> Optional[datetime]:
    try:
        with Image.open(image_path) as img:
            exif_data = img._getexif()
            if exif_data:
                exif = {ExifTags.TAGS.get(tag, tag): value for tag, value in exif_data.items()}
                date_str = exif.get("DateTimeOriginal") or exif.get("DateTime")
                if date_str:
                    return datetime.strptime(date_str, "%Y:%m:%d %H:%M:%S")
    except Exception as e:
        print(f"[Warning] Could not read EXIF from {image_path}: {e}")
    return None

def get_creation_date(file_path: Path) -> datetime:
    exif_date = get_exif_creation_date(file_path)
    if exif_date:
        return exif_date
    return datetime.fromtimestamp(file_path.stat().st_ctime)

# ---------------------------
# File Identification & Hashing Functions
# ---------------------------
def is_allowed_file(filepath: Path) -> bool:
    lower_name = filepath.name.lower()
    for ext in ALLOWED_EXTENSIONS:
        pattern = re.compile(rf"\.{ext[1:]}(\d+)?$")
        if pattern.search(lower_name):
            return True
    return False

def get_file_hash(filepath: Path) -> str:
    hash_func = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    return hash_func.hexdigest()

def get_image_phash(filepath: Path) -> Optional[str]:
    try:
        with Image.open(filepath) as img:
            return str(imagehash.phash(img))
    except (UnidentifiedImageError, OSError):
        return None

# ---------------------------
# Grouping Key Extraction Functions
# ---------------------------
def get_group_folder(filename: str) -> str:
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

def composite_group_key(file: Path) -> str:
    group_key = get_group_folder(file.name)
    creation_date = get_creation_date(file)
    date_str = creation_date.strftime("%Y-%m-%d")
    if group_key == "Miscellaneous":
        return f"Miscellaneous_{date_str}"
    elif group_key[0].isdigit():
        return group_key
    else:
        return f"{group_key}_{date_str}"

# ---------------------------
# Content Check via pHash
# ---------------------------
def check_content_similarity(target_folder: Path, file: Path, threshold: int = PHASH_THRESHOLD) -> bool:
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
# Curated Selection Function
# ---------------------------
def curate_group(files: List[Path]) -> Optional[Path]:
    best_score = -1.0
    best_file = None
    for file in files:
        score = compute_aesthetic_score_advanced(file)
        print(f"  {file.name}: Aesthetic score = {score:.2f}")
        if score > best_score:
            best_score = score
            best_file = file
    return best_file

# ---------------------------
# Main Workflow: Grouping into Subfolders and Curation (Test on First 20 Images)
# ---------------------------
def main():
    source = Path(SOURCE_DIR)
    organized = Path(ORGANIZED_DIR)
    curated = Path(CURATED_DIR)
    organized.mkdir(parents=True, exist_ok=True)
    curated.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Recursively gather allowed photo files from the source folder
    image_files = [f for f in source.rglob("*") if f.is_file() and is_allowed_file(f)]
    print(f"Found {len(image_files)} images in source.")
    
    # Process only the first 20 images for testing
    test_files = image_files[:50]
    print(f"Processing first {len(test_files)} images for test.")
    
    # Step 2: Group images into subfolders in ORGANIZED_DIR based on composite key (filename + creation date)
    groups: Dict[str, List[Path]] = {}
    for file in test_files:
        key = composite_group_key(file)
        groups.setdefault(key, []).append(file)
    print(f"Formed {len(groups)} groups based on naming and creation date.")
    
    # Step 3: For each group, move images into subfolders in ORGANIZED_DIR.
    org_log_entries = []
    for group_key, files in groups.items():
        target_folder = organized / group_key
        target_folder.mkdir(exist_ok=True)
        for file in files:
            dest_path = target_folder / file.name
            counter = 1
            while dest_path.exists():
                stem, ext = os.path.splitext(file.name)
                dest_path = target_folder / f"{stem}_{counter}{ext}"
                counter += 1
            shutil.copy2(file, dest_path)
            org_log_entries.append((file.name, str(file), str(dest_path), group_key, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            print(f"Moved '{file.name}' to folder '{group_key}'")
    
    # Write organization log CSV
    with open("organization_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "original_path", "new_path", "group_key", "timestamp"])
        writer.writerows(org_log_entries)
    print("Organization log saved to organization_log.csv")
    
    # Step 4: For each group subfolder in ORGANIZED_DIR, curate the best photo and copy it to CURATED_DIR.
    curated_log = []
    for group_folder in organized.iterdir():
        if group_folder.is_dir():
            group_images = [f for f in group_folder.iterdir() if f.is_file() and is_allowed_file(f)]
            if not group_images:
                continue
            print(f"\nCurating group '{group_folder.name}' with {len(group_images)} images:")
            best_image = curate_group(group_images)
            if best_image:
                creation_date = get_creation_date(best_image)
                date_folder = creation_date.strftime("%Y-%m")
                target_curated_folder = curated / f"{group_folder.name}_{date_folder}"
                target_curated_folder.mkdir(parents=True, exist_ok=True)
                dest_curated = target_curated_folder / best_image.name
                counter = 1
                while dest_curated.exists():
                    stem, ext = os.path.splitext(best_image.name)
                    dest_curated = target_curated_folder / f"{stem}_{counter}{ext}"
                    counter += 1
                shutil.copy2(best_image, dest_curated)
                curated_log.append({
                    "group": group_folder.name,
                    "filename": best_image.name,
                    "original_path": str(best_image),
                    "curated_path": str(dest_curated),
                    "aesthetic_score": f"{compute_aesthetic_score_advanced(best_image):.2f}",
                    "created_at": creation_date.strftime("%Y-%m-%d %H:%M:%S")
                })
                print(f"  Curated '{best_image.name}' to folder '{target_curated_folder.name}'")
    
    # Write curated log CSV
    curated_csv = Path("curated_log.csv")
    with open(curated_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "filename", "original_path", "curated_path", "aesthetic_score", "created_at"])
        writer.writeheader()
        for row in curated_log:
            writer.writerow(row)
    print(f"\nCuration complete. Log saved to {curated_csv}")

if __name__ == "__main__":
    main()
