import numpy as np
import matplotlib.pyplot as plt
import json
import os
import shutil
import math
from pathlib import Path
from cell_config import cell, cell_ls


def radians_to_degrees(radians):
    """Convert radians to degrees and normalize to 0-360 range."""
    degrees = math.degrees(radians)
    # Normalize to 0-360 range
    degrees = degrees % 360
    if degrees < 0:
        degrees += 360
    return degrees


def round_to_nearest_10(degrees):
    """Round degrees to nearest 10 degrees."""
    return round(degrees / 10) * 10


def is_point_in_cell(x, y, cell_obj):
    """Check if point (x, y) is within the rectangular bounds of a cell."""
    return (cell_obj.x_min <= x <= cell_obj.x_max and 
            cell_obj.y_min <= y <= cell_obj.y_max)


def organize_json_files(data_dir="data", output_base_dir="cells_kernels"):
    """
    Organize JSON files from data directory to cells_kernels based on position and orientation.
    
    For each JSON file:
    1. Extract position (x, y) and orientation (in radians)
    2. Check if position is within any cell's rectangular region
    3. If yes, convert orientation to degrees, round to nearest 10 degrees
    4. Copy file to cells_kernels/c{cell_index}/deg{rounded_degrees}/
    """
    data_path = Path(data_dir)
    output_base_path = Path(output_base_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return
    
    # Get all JSON files in data directory
    json_files = list(data_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process.")
    
    processed_count = 0
    skipped_count = 0
    
    for json_file in json_files:
        try:
            # Load JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract position (x, y) and orientation
            position = data.get("position", [])
            orientation = data.get("orientation", [])
            
            if len(position) < 2 or len(orientation) < 1:
                print(f"Skipping {json_file.name}: missing position or orientation data")
                skipped_count += 1
                continue
            
            x, y = position[0], position[1]
            orientation_rad = orientation[0]
            
            # Check if point is within any cell's bounds
            found_cell = False
            for cell_idx, cell_obj in enumerate(cell_ls):
                if is_point_in_cell(x, y, cell_obj):
                    found_cell = True
                    
                    # Convert orientation from radians to degrees and round to nearest 10
                    degrees = radians_to_degrees(orientation_rad)
                    rounded_degrees = int(round_to_nearest_10(degrees))
                    
                    # Ensure it's in 0-360 range (handle edge case where rounding gives 360)
                    if rounded_degrees == 360:
                        rounded_degrees = 0
                    
                    # Create destination path: cells_kernels/c{cell_idx}/deg{rounded_degrees}/
                    dest_dir = output_base_path / f"c{cell_idx}" / f"deg{rounded_degrees}"
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Copy file to destination
                    dest_file = dest_dir / json_file.name
                    shutil.copy2(json_file, dest_file)
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"Processed {processed_count} files...")
                    
                    break  # Only copy to first matching cell
            
            if not found_cell:
                skipped_count += 1
                if skipped_count % 100 == 0:
                    print(f"Skipped {skipped_count} files (not in any cell)...")
        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped: {skipped_count} files")


def organize_by_orientation(data_dir="data", output_dir="data_deg"):
    """
    Organize JSON files from data directory to data_deg based on orientation.
    
    For each JSON file:
    1. Extract orientation (in radians, range -pi to pi)
    2. Map to nearest cardinal direction: deg0, deg90, deg180, or deg270
    3. Copy file to data_deg/deg{0|90|180|270}/
    
    Orientation mapping (radians to degrees):
    - -pi/4 to pi/4: deg0 (pointing right/east)
    - pi/4 to 3pi/4: deg90 (pointing up/north)
    - 3pi/4 to -3pi/4 (wrapping): deg180 (pointing left/west)
    - -3pi/4 to -pi/4: deg270 (pointing down/south)
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    
    if not data_path.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create the four degree folders
    for deg in [0, 90, 180, 270]:
        (output_path / f"deg{deg}").mkdir(exist_ok=True)
    
    # Get all JSON files in data directory
    json_files = list(data_path.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process.")
    
    processed_count = 0
    skipped_count = 0
    
    for json_file in json_files:
        try:
            # Load JSON file
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract orientation
            orientation = data.get("orientation", [])
            
            if len(orientation) < 1:
                print(f"Skipping {json_file.name}: missing orientation data")
                skipped_count += 1
                continue
            
            orientation_rad = orientation[0]
            
            # Map orientation (in radians, -pi to pi) to nearest cardinal direction
            # -pi/4 to pi/4: deg0 (pointing right/east)
            # pi/4 to 3pi/4: deg90 (pointing up/north)
            # 3pi/4 to -3pi/4 (wrapping): deg180 (pointing left/west)
            # -3pi/4 to -pi/4: deg270 (pointing down/south)
            
            if -math.pi/4 <= orientation_rad <= math.pi/4:
                target_deg = 0
            elif math.pi/4 < orientation_rad <= 3*math.pi/4:
                target_deg = 90
            elif orientation_rad > 3*math.pi/4 or orientation_rad <= -3*math.pi/4:
                target_deg = 180
            else:  # -3*math.pi/4 < orientation_rad < -math.pi/4
                target_deg = 270
            
            # Create destination path: data_deg/deg{target_deg}/
            dest_dir = output_path / f"deg{target_deg}"
            
            # Copy file to destination
            dest_file = dest_dir / json_file.name
            shutil.copy2(json_file, dest_file)
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files...")
        
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"  Processed: {processed_count} files")
    print(f"  Skipped: {skipped_count} files")


if __name__ == "__main__":
    organize_json_files(data_dir='data')
    # organize_by_orientation(data_dir="data", output_dir="data_deg")











