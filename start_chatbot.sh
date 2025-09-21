#!/bin/bash

echo "🌍 Starting Travel Planner Chatbot..."
echo "======================================"

# Check if we're in the right directory
if [ ! -f "streamlit_travel_planner.py" ]; then
    echo "❌ Error: Please run this script from the streamlit-travel-planer folder"
    echo "   Navigate to: /Users/virgilijus/Desktop/Projects/streamlit-travel-planer"
    echo "   Then run: ./start_chatbot.sh"
    exit 1
fi

# Check if requirements are installed
echo "📦 Checking requirements..."
if ! python3 -c "import streamlit, google.genai" 2>/dev/null; then
    echo "📦 Installing required packages..."
    pip3 install streamlit google-genai
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install packages. Please check your internet connection."
        echo "   You can manually install with: pip3 install streamlit google-genai"
        exit 1
    fi
fi

echo "✅ All requirements satisfied!"
echo ""
echo "🚀 Starting Travel Planner Chatbot..."
echo "   Your web browser will open automatically"
echo "   If not, go to: http://localhost:8501"
echo ""
echo "💡 Remember to get your API key from:"
echo "   https://aistudio.google.com/app/apikey"
echo ""
echo "🛑 To stop the chatbot, press Ctrl+C in this window"
echo ""

# Start the Streamlit app
streamlit run streamlit_travel_planner.py
