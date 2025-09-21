# 🌍 Travel Planner Chatbot

A user-friendly travel planning application that helps you create personalized itineraries using Google AI Studio. Available in both web interface and command-line versions!

## 🌟 Features

- **Web Interface**: Easy-to-use chatbot with beautiful interface (Recommended for beginners)
- **Command Line**: Traditional terminal-based application
- **Personalized Itineraries**: Get custom travel plans based on your interests
- **Flexible Duration**: Plan trips from 1 day to several weeks
- **Adjustable Detail Level**: Choose between short or long itinerary formats
- **Activity-Based Recommendations**: Tailored suggestions based on your preferred activities
- **Download Options**: Save your itinerary to a text file for future reference

## 📋 What You Need

- MacBook Air (macOS)
- Python 3.8 or higher
- Google AI Studio API key (free to get)
- Internet connection

## 🚀 Quick Start Guide (For Non-Programmers)

### Step 1: Get Your Google AI Studio API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the generated API key (it starts with "AIza...")
5. Keep this key safe - you'll need it to run the application

### Step 2: Setup Your Application

1. Open **Terminal** (find it in Applications > Utilities)
2. Navigate to the project folder by typing:
   ```bash
   cd /Users/virgilijus/Desktop/Projects/streamlit-travel-planer
   ```
3. Run the setup script:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

### Step 3: Run the Travel Planner

#### Option A: Web Interface (Recommended for Beginners)

1. In Terminal, type:
   ```bash
   streamlit run streamlit_travel_planner.py
   ```
2. Your web browser will automatically open with the Travel Planner
3. Enter your API key in the sidebar
4. Fill in your travel preferences:
   - **Country**: Where you want to visit (e.g., Italy)
   - **Activities**: What you enjoy (e.g., Hiking, Cooking)
   - **Days**: How many days for your trip (e.g., 3)
   - **Detail Level**: Short or Long itinerary
5. Click "Generate My Travel Itinerary"
6. Download your itinerary when ready!

#### Option B: Command Line Interface

1. In Terminal, type:
   ```bash
   python3 planner.py
   ```
2. Follow the prompts to enter your preferences

## 📝 Example Usage

### Input Example:
- **Country**: Italy
- **Activities**: Hiking, Cooking, Art Museums
- **Days**: 5
- **Detail Level**: Long

### Output Example:
The application will generate a comprehensive 5-day itinerary for Italy including:
- Daily schedules with hiking trails
- Cooking classes and food experiences
- Art museum recommendations
- Transportation tips
- Budget estimates
- Best time to visit advice

## 🔧 Troubleshooting

### "Command not found" errors:
- Make sure Python 3 is installed: `python3 --version`
- Install Python from [python.org](https://www.python.org/downloads/) if needed

### API Key issues:
- Ensure your API key starts with "AIza"
- Check you have internet connection
- Verify the API key is copied correctly from Google AI Studio

### Package installation issues:
- Try: `pip3 install --upgrade pip` first
- Then run: `pip3 install -r requirements.txt`

## 📱 Screenshots

The web interface includes:
- Sidebar for API key input
- Clean form for travel preferences
- Real-time itinerary generation
- Download functionality
- Mobile-responsive design

## 🤝 Support

If you need help:
1. Check the troubleshooting section above
2. Ensure all steps in the setup guide are completed
3. Verify your internet connection and API key

## 📄 Files in This Project

- `streamlit_travel_planner.py` - Web interface application
- `planner.py` - Command line application
- `requirements.txt` - List of required packages
- `setup.sh` - Automatic setup script
- `README.md` - This instruction file

---

**Happy travels! 🎉✈️**
```

### Step 4: Run the Application

In Terminal, navigate to the project folder and run:

```bash
cd /Users/virgilijus/Desktop/Projects/streamlit-travel-planer
python3 planner.py
```

**The application will ask you to enter your API key when it starts!** 🔑

## 🎯 How to Use

1. **Run the application** using the command above
2. **Enter your Google AI Studio API key** when prompted (starts with "AIza...")
3. **Enter your destination** (e.g., "Italy", "Japan", "France")
4. **Specify your interests** (e.g., "Hiking, Cooking", "Museums, Shopping")
5. **Choose trip duration** (number of days)
6. **Select detail level** ("short" for concise or "long" for detailed)
7. **Wait for generation** (usually takes 10-30 seconds)
8. **Review your itinerary** displayed on screen
9. **Optionally save** the itinerary to a text file

## 📝 Example Usage

```
🌍 Welcome to the Travel Planner!
==================================================

🔑 Google AI Studio API Key Required
========================================
You can get your API key from: https://aistudio.google.com/app/apikey
The API key will look like: AIza...

Please enter your Google AI Studio API key: AIza...your_key_here...
🔄 Connecting to Google AI Studio...
✅ Successfully connected to Google AI Studio!

📝 Travel Planning Details
========================================
Which country would you like to visit? (e.g., Italy): Italy
What activities do you enjoy? (e.g., Hiking, Cooking): Hiking, Photography
How many days will you spend there? 5
Would you like a 'long' or 'short' itinerary? long

🤖 Generating your personalized travel itinerary...
This may take a few moments...

============================================================
🎉 YOUR PERSONALIZED TRAVEL ITINERARY
============================================================
[Generated itinerary will appear here]
```

## ❌ Troubleshooting

### "API key validation failed" or "Unable to connect"
- Make sure you copied the complete API key from Google AI Studio
- Verify the API key starts with "AIza"
- Check that your API key has proper permissions and quota
- Ensure you have internet connection

### "No module named 'google'"
- Install the required package: `pip3 install google-generativeai`
- If still having issues, try: `pip3 install --upgrade google-generativeai`

### "Permission denied" when running python3
- Try: `sudo python3 planner.py`
- Or check file permissions: `chmod +x planner.py`

### Application runs but generates no itinerary
- Check your internet connection
- Verify your API key is correct and has quota remaining
- Try again in a few minutes (API might be temporarily busy)

## 🔧 Advanced Options

### Running with Different Models
The application uses `gemini-2.0-flash-001` by default. This is currently the best model for this task.

### Customizing Output
You can modify the `create_travel_prompt()` function in `planner.py` to add more specific requirements or change the output format.

## 💡 Tips for Best Results

1. **Be specific with activities**: Instead of "sightseeing", try "historical sites, local markets"
2. **Realistic duration**: Plan 2-7 days for best detailed itineraries
3. **Clear location**: Use country names or major cities for better results
4. **Try both formats**: Short format for quick overview, long format for detailed planning

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all installation steps were followed correctly
3. Verify your API key is working at [Google AI Studio](https://aistudio.google.com/)

## 🎉 Enjoy Your Trip Planning!

The application will generate personalized itineraries based on your preferences. Have fun exploring new destinations!
