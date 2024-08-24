#!/bin/bash

# Set the source and destination directories
SRC_DIR="/data_ephemeral/rahul/models/Meta-Llama-3.1-405B/"
DEST_DIR="/scratch/rahul/models/Meta-Llama-3.1-405B/"

# Number of parallel processes (adjust based on your system's capabilities)
PARALLEL_JOBS=128

# Find all files in the source directory and pipe them to parallel
find "$SRC_DIR" -type f | parallel -j $PARALLEL_JOBS rsync -avzh {} "$DEST_DIR"