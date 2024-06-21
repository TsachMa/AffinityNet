#!/bin/bash

# Define the directory to search in
directory="/Users/tsachmackey/dfs/affinity_net/PDBbind"

# Define the log file
log_file="error.log"

# Find all files ending with "_protein.pdb" recursively
files=$(find "$directory" -type f -name "*_protein.pdb")
total_files=$(echo "$files" | wc -l)
converted_files=0
start_time=$(date +%s)

# Process each file with obabel
while IFS= read -r file; do
    # Remove the .pdb extension if it exists
    file_no_ext="${file%.pdb}"
    # Process the file with obabel and redirect stderr to the log file
    obabel "$file" -O "$file_no_ext.mol2" -p 7 2>> "$log_file"
    if [ $? -ne 0 ]; then
        echo "Error processing file: $file" >> "$log_file"
    fi
    converted_files=$((converted_files + 1))
    # Print total number of files processed
    echo "Total files processed: $converted_files"
done <<< "$files"
