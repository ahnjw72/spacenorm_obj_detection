#!/bin/bash

count_files() {
    local dir="$1"
    
    # Iterate through all files and directories in the given directory
    for entry in "$dir"/*; do
        if [[ -f "$entry" ]]; then
            # Print the file count for the current subdirectory
            echo "$(basename "$dir"): $(find "$dir" -maxdepth 1 -type f | wc -l)"
            break
        elif [[ -d "$entry" ]]; then
            # Recursively call the function for subdirectories
            count_files "$entry"
        fi
    done
}

# Prompt the user for the directory to start the count from
read -p "Enter the directory path: " directory

# Call the function with the provided directory
count_files "$directory"
