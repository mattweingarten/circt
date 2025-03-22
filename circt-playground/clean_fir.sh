#!/bin/bash

# Check if a file is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Use sed to delete matching lines
sed -E '/when.*@\[/ {N; /printf.*@\[/d;}' "$1"
