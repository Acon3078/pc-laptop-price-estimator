#!/bin/bash
# Setup script to install dependencies and run cleaning pipeline

echo "Setting up Python environment for data cleaning..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing required packages..."
pip install -q pandas numpy matplotlib seaborn

# Run cleaning script
echo ""
echo "Running cleaning pipeline..."
python3 run_cleaning.py

echo ""
echo "Done! Check data/clean/db_computers_cleaned.csv for the cleaned dataset."

