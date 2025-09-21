#!/bin/bash

echo "🌍 Travel Planner Setup Script"
echo "================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 from https://www.python.org/downloads/"
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Install required packages
echo "📦 Installing required packages..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ All packages installed successfully!"
else
    echo "❌ Failed to install packages. Please check your internet connection."
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To start using the Travel Planner Chatbot:"
echo ""
echo "Option 1 - Web Interface (Recommended for beginners):"
echo "   Run: streamlit run streamlit_travel_planner.py"
echo "   This will open a web browser with an easy-to-use interface"
echo ""
echo "Option 2 - Command Line Interface:"
echo "   Run: python3 planner.py"
echo "   This runs in the terminal"
echo ""
echo "📝 Note: You'll need your Google AI Studio API key"
echo "   Get it from: https://aistudio.google.com/app/apikey"
echo ""
echo "🎉 Happy travel planning!"
