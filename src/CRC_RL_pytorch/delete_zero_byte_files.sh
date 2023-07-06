#!/bin/bash

# This script will delete all files in the current directory and its subdirectories
# that have a size of 0 bytes.

# Check if the du command is available.
if ! command -v du >/dev/null; then
  echo "The du command is not available."
  exit 1
fi

# Get the list of all files in the current directory and its subdirectories.
files=$(find . -type f)

# Loop over the list of files and delete any files that have a size of 0 bytes.
for file in $files; do
    #if du -hs $file | grep -q 0; then
    if du -hs $file | grep -P "\b0\b"; then
        #echo "Deleting file $file."
        echo $file
        #rm $file
    fi
done