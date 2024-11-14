#!/bin/bash

# Check if directory argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <directory>"
    echo "Example: $0 /path/to/ply/files"
    exit 1
fi

# Check if directory exists
if [ ! -d "$1" ]; then
    echo "Error: Directory '$1' does not exist"
    exit 1
fi

# Store original directory
ORIGINAL_DIR=$(pwd)

# Change to target directory
cd "$1" || exit 1
echo "Working in directory: $(pwd)"

# Function to display string context
show_context() {
    echo "File content around potential match:"
    # Show binary content around the area
    xxd "$1" | grep -A 2 -B 2 -i "scalar" || true
    echo "---"
    # Alternative view with strings
    strings "$1" | grep -A 2 -B 2 -i "scalar" || true
}

# Function to check if string exists in file (more thorough)
check_string() {
    local file="$1"
    echo "Checking file: $file"
    
    # Try different methods to find the string
    if strings "$file" | grep -q "scalar_Classification"; then
        echo "Found using strings command"
        return 0
    elif xxd "$file" | grep -q "scalar_Classification"; then
        echo "Found using xxd command"
        return 0
    elif hexdump -C "$file" | grep -q "scalar_Classification"; then
        echo "Found using hexdump command"
        return 0
    else
        echo "String not found with any method"
        show_context "$file"
        return 1
    fi
}

# Function to replace string in file
replace_string() {
    local file="$1"
    local temp_file="${file}.temp"
    
    # Make backup if it doesn't exist
    if [ ! -f "${file}.backup" ]; then
        cp "$file" "${file}.backup"
        echo "Created backup: ${file}.backup"
    fi
    
    echo "Attempting replacement..."
    # Try perl binary replacement
    perl -p -e 's/scalar_Classification/classification      /g' < "$file" > "$temp_file"
    
    # Verify replacement
    if check_string "$temp_file"; then
        echo "Warning: String still exists after replacement attempt"
        rm "$temp_file"
        return 1
    else
        mv "$temp_file" "$file"
        echo "Replacement successful"
        return 0
    fi
}

# Check if any PLY files exist
if ! ls *.ply >/dev/null 2>&1; then
    echo "No PLY files found in directory: $1"
    cd "$ORIGINAL_DIR" || exit 1
    exit 1
fi

# Main loop
for file in *.ply; do
    if [ -f "$file" ]; then
        echo "=== Processing: $file ==="
        if check_string "$file"; then
            echo "Found string in $file"
            if replace_string "$file"; then
                echo "Successfully processed $file"
            else
                echo "Failed to process $file"
            fi
        else
            echo "String not found in $file"
        fi
        echo "=== Done with $file ==="
        echo
    fi
done

# Return to original directory
cd "$ORIGINAL_DIR" || exit 1

echo "Process complete. Backups have been created with .backup extension"