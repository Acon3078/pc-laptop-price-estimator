#!/bin/bash
# Setup script to install dependencies for the project

echo "Setting up Python environment..."

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
pip install -q -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To run data cleaning and model training:"
echo "  - Open Model_Training.ipynb in Jupyter/VS Code"
echo "  - Run all cells to generate cleaned dataset and model artifacts"
echo ""
echo "To run the Streamlit app:"
echo "  streamlit run app.py"

