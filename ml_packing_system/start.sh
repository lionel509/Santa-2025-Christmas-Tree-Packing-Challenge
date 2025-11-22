#!/bin/bash

# Quick start script for Santa 2025 ML Packing System

echo "================================================"
echo "Santa 2025 - ML Packing System"
echo "================================================"
echo ""

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8 or higher."
    exit 1
fi

# Check if in correct directory
if [ ! -f "run.py" ]; then
    echo "❌ Error: run.py not found. Please run this script from the ml_packing_system directory."
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
pip install -q -r requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"
echo ""

# Run the system
echo "🚀 Starting ML Packing System..."
echo ""
echo "The system will:"
echo "  1. Initialize/load puzzle states"
echo "  2. Start autonomous optimization"
echo "  3. Launch web interface on http://127.0.0.1:8000/app"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python3 run.py "$@"
