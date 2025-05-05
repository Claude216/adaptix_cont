# this files is used to merge json files into one corpus file (still json file)
# it requires the folder path, the target size of the output corpus file as input argument

import os
import json
import argparse
from pathlib import Path

def merge_json_files(input_dir, output_file, max_size_mb=500):
    """
    Merge JSON files from input_dir into a single file until it reaches max_size_mb.
    
    Args:
        input_dir (str): Directory containing JSON files
        output_file (str): Path to output merged JSON file
        max_size_mb (int): Maximum size of the merged file in MB
    
    Returns:
        int: Number of files processed
    """
    # Convert max_size to bytes
    max_size_bytes = max_size_mb * 1024 * 1024
    
    # Get all JSON files in the directory
    json_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.json')]
    print(f"Found {len(json_files)} JSON files in {input_dir}")
    
    # Initialize output as an empty list to store all JSON objects
    merged_data = []
    
    # Track processed files and current file size
    processed_files = 0
    current_size = 0
    
    # Process files one by one
    for json_file in json_files:
        file_path = os.path.join(input_dir, json_file)
        
        try:
            # Read the current JSON file
            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = json.load(f)
            
            # Add the data to our merged_data list
            if isinstance(file_data, list):
                merged_data.extend(file_data)
            else:
                merged_data.append(file_data)
            
            # Write to the output file to check its size
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f)
            
            # Get the current file size
            current_size = os.path.getsize(output_file)
            processed_files += 1
            
            print(f"Processed {processed_files}/{len(json_files)} files. Current size: {current_size/1024/1024:.2f}MB")
            
            # Check if we've reached the size limit
            if current_size >= max_size_bytes:
                print(f"Reached size limit of {max_size_mb}MB. Stopping.")
                break
                
        except json.JSONDecodeError:
            print(f"Error: Could not parse {json_file} as JSON. Skipping.")
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}. Skipping.")
    
    return processed_files

def main():
    parser = argparse.ArgumentParser(description='Merge JSON files until reaching a specified size limit')
    parser.add_argument('input_dir', help='Directory containing JSON files to merge')
    parser.add_argument('output_file', help='Output merged JSON file')
    parser.add_argument('--max-size', type=int, default=500, help='Maximum size in MB (default: 500)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_file)
    output_dir = output_path.parent
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    
    # Process files
    num_processed = merge_json_files(args.input_dir, args.output_file, args.max_size)
    
    print(f"\nMerge complete. Processed {num_processed} files.")
    print(f"Output file: {args.output_file}")
    print(f"Final size: {os.path.getsize(args.output_file)/1024/1024:.2f}MB")

if __name__ == "__main__":
    main()