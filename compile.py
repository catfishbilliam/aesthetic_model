import os
import shutil
import difflib
from pathlib import Path

SOURCE_DIR = "/Users/wconway/Library/CloudStorage/Box-Box/cumulus_transfer"
SIMILARITY_THRESHOLD = 0.7  # Adjust this to be more/less strict

def get_similarity(a, b):
    """Return a ratio (0 to 1) of how similar two strings are."""
    return difflib.SequenceMatcher(None, a, b).ratio()

def find_cluster(folder_name, clusters):
    """
    Given a folder_name and the current clusters (list of [representative, [members]]),
    return the index of the cluster if folder_name is similar enough to the representative.
    Otherwise return None.
    """
    for idx, (rep_name, members) in enumerate(clusters):
        similarity = get_similarity(folder_name.lower(), rep_name.lower())
        if similarity >= SIMILARITY_THRESHOLD:
            return idx
    return None

def main():
    source = Path(SOURCE_DIR)
    if not source.is_dir():
        print(f"Source directory does not exist: {source}")
        return

    # 1) Gather all subfolders (ignore files).
    all_subfolders = [d for d in source.iterdir() if d.is_dir()]

    # 2) Build clusters based on name similarity.
    #    Each cluster is (representative_folder_name, [list_of_folder_paths]).
    #    We pick the first folder in a cluster as the representative.
    clusters = []

    for folder_path in all_subfolders:
        folder_name = folder_path.name
        idx = find_cluster(folder_name, clusters)
        if idx is not None:
            # Add to existing cluster
            clusters[idx][1].append(folder_path)
        else:
            # Create new cluster with this folder as representative
            clusters.append([folder_name, [folder_path]])

    # 3) For each cluster, create a single merged folder, move contents, delete originals
    for rep_name, members in clusters:
        # Decide on a name for the merged folder (e.g., the representative's name)
        merged_folder_name = f"{rep_name}_merged"
        merged_folder = source / merged_folder_name

        # If it already exists, adjust the name
        counter = 1
        while merged_folder.exists():
            merged_folder = source / f"{merged_folder_name}_{counter}"
            counter += 1

        merged_folder.mkdir()

        print(f"\n=== Merging {len(members)} folder(s) into: {merged_folder.name}")
        for m_folder in members:
            print(f"  Moving contents of '{m_folder.name}' → '{merged_folder.name}'")
            for item in m_folder.iterdir():
                # If there's a naming collision, rename or skip as needed
                dest = merged_folder / item.name
                dest_count = 1
                while dest.exists():
                    # For collisions, rename with a counter
                    stem = dest.stem
                    suffix = dest.suffix
                    dest = merged_folder / f"{stem}_{dest_count}{suffix}"
                    dest_count += 1

                if item.is_dir():
                    shutil.move(str(item), str(dest))
                else:
                    shutil.move(str(item), str(dest))

            # Delete the now-empty original folder
            print(f"  Deleting empty folder: {m_folder.name}")
            m_folder.rmdir()

    print("\nDone! All similar folders have been merged and originals deleted.")

if __name__ == "__main__":
    main()
