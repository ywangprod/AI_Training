#!/bin/bash

# Security Review Setup Script

echo "Setting up Security Code Review System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv security-review-venv
source security-review-venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install openai requests

# Make the security reviewer executable
chmod +x security_reviewer.py

echo ""
echo "Setup complete! Next steps:"
echo "1. Get an OpenAI API key from https://platform.openai.com/api-keys"
echo "2. Set your API key: export OPENAI_API_KEY='your-api-key-here'"
echo "3. Run the security review: python security_reviewer.py /path/to/your/code"
echo ""
echo "You can also run: ./security_reviewer.py /path/to/your/code"