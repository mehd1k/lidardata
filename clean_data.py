import json
import os
import math
from pathlib import Path
from collections import defaultdict


def euclidean_distance(pos1, pos2):
    """Calculate Euclidean distance between two 2D positions."""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def clean_redundant_data(directory, distance_threshold=0.1):
    """
    Remove redundant JSON files from a directory.
    
    For files with positions close to each other (within distance_threshold),
    keeps only the oldest one (based on timestamp).
    
    Args:
        directory: Path to directory containing JSON files
        distance_threshold: Maximum distance between positions to consider them "close" (default: 0.1)
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist.")
        return
    
    if not dir_path.is_dir():
        print(f"Error: '{directory}' is not a directory.")
        return
    
    # Get all JSON files in directory
    json_files = list(dir_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process.")
    
    if len(json_files) == 0:
        print("No JSON files found.")
        return
    
    # Load all files with their positions and timestamps
    file_data = []
    invalid_files = []
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            position = data.get("position", [])
            timestamp = data.get("timestamp", None)
            
            if len(position) < 2 or timestamp is None:
                invalid_files.append(json_file)
                continue
            
            x, y = position[0], position[1]
            file_data.append({
                'file': json_file,
                'position': (x, y),
                'timestamp': timestamp
            })
        
        except Exception as e:
            print(f"Error reading {json_file.name}: {e}")
            invalid_files.append(json_file)
    
    if len(invalid_files) > 0:
        print(f"Skipped {len(invalid_files)} files with invalid data.")
    
    if len(file_data) == 0:
        print("No valid JSON files found.")
        return
    
    print(f"Processing {len(file_data)} valid files...")
    
    # Group files by proximity
    # Use a simple approach: for each file, find all nearby files
    files_to_keep = set()
    files_to_delete = []
    
    # Sort by timestamp (oldest first) to prioritize keeping older files
    file_data.sort(key=lambda x: x['timestamp'])
    
    # For each file, check if there's an older file within the distance threshold
    for i, current_file in enumerate(file_data):
        should_keep = True
        
        # Check all files that came before (older timestamps)
        for j, other_file in enumerate(file_data[:i]):
            # If we already decided to keep this other file, check distance
            if other_file['file'] in files_to_keep:
                distance = euclidean_distance(current_file['position'], other_file['position'])
                if distance <= distance_threshold:
                    # There's an older file nearby, so we should delete this one
                    should_keep = False
                    break
        
        if should_keep:
            files_to_keep.add(current_file['file'])
        else:
            files_to_delete.append(current_file['file'])
    
    # Delete redundant files
    deleted_count = 0
    for file_to_delete in files_to_delete:
        try:
            file_to_delete.unlink()
            deleted_count += 1
        except Exception as e:
            print(f"Error deleting {file_to_delete.name}: {e}")
    
    print(f"\nCleaning complete!")
    print(f"  Files kept: {len(files_to_keep)}")
    print(f"  Files deleted: {deleted_count}")
    print(f"  Distance threshold used: {distance_threshold}")


if __name__ == "__main__":
    import sys
    
    # Get directory from command line argument or use default
    # if len(sys.argv) > 1:
    directory = "cells_kernels/c1/deg0"
    
    # Get distance threshold from command line or use default
    distance_threshold = 0.1
    if len(sys.argv) > 2:
        try:
            distance_threshold = float(sys.argv[2])
        except ValueError:
            print(f"Warning: Invalid distance threshold '{sys.argv[2]}', using default 0.1")
    
    clean_redundant_data(directory, distance_threshold)
