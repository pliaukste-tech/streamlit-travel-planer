# 🚀 Quick Start Guide - Travel Planner Chatbot

## Simple Instructions for Non-Programmers

### What This Does
This chatbot helps you plan travel itineraries by asking for:
- **Country** (e.g., Italy)
- **Activities you like** (e.g., Hiking, Cooking)
- **Days to spend** (e.g., 3)
- **Report length** (Long or Short)

You enter your Google AI Studio API key, and it creates a personalized travel plan!

---

## How to Run the Chatbot

### Step 1: Get Your API Key
1. Go to: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key (starts with "AIza...")

### Step 2: Open Terminal
1. Press `Cmd + Space`
2. Type "Terminal" and press Enter

### Step 3: Go to Your Project Folder
Copy and paste this into Terminal:
```bash
cd /Users/virgilijus/Desktop/Projects/streamlit-travel-planer
```

### Step 4: Install Requirements (First Time Only)
Copy and paste this:
```bash
pip3 install streamlit google-genai
```

### Step 5: Start the Chatbot
Copy and paste this:
```bash
streamlit run streamlit_travel_planner.py
```

### Step 6: Use the Chatbot
1. A web browser will open automatically
2. Enter your API key in the sidebar (left side)
3. Fill in the form:
   - Country: Italy
   - Activities: Hiking, Cooking
   - Days: 3
   - Detail: short or long
4. Click "Generate My Travel Itinerary"
5. Download your itinerary when ready!

---

## Alternative: Command Line Version

If you prefer the original command-line version:
```bash
python3 planner.py
```

---

## Troubleshooting

**If you get "command not found":**
- Install Python from: https://www.python.org/downloads/

**If API key doesn't work:**
- Make sure it starts with "AIza"
- Check you copied it completely

**If packages don't install:**
```bash
pip3 install --upgrade pip
pip3 install streamlit google-genai
```

---

**That's it! Enjoy planning your trips! 🎉✈️**

## How to Use the Application

1. **API Key**: The app will ask for your Google AI key first - paste it in
2. **Country**: Type the country you want to visit (e.g., "Italy")
3. **Activities**: Type what you like to do (e.g., "hiking, food tours")
4. **Days**: Type how many days you'll be there (e.g., "5")
5. **Detail**: Type "short" for a quick plan or "long" for detailed plan

Example:
```
Please enter your Google AI Studio API key: AIza...your_key_here...
Which country would you like to visit? Italy
What activities do you enjoy? hiking, cooking classes
How many days will you spend there? 4
Would you like a 'long' or 'short' itinerary? long
```

The AI will then create a custom travel plan for you!

## If Something Goes Wrong

### "Command not found" error
- Make sure you're in the right folder
- Try typing the full path: `/Users/virgilijus/Desktop/Projects/streamlit-travel-planer/`

### "API key validation failed" error
- Make sure you copied the entire API key correctly
- Check that it starts with "AIza"
- Verify your internet connection
- Make sure the API key has permissions in Google AI Studio

### Can't find Terminal
- Use Spotlight (Cmd + Space) and search for "Terminal"
- Or go to Applications > Utilities > Terminal

### Need Help?
- Make sure you followed each step exactly
- Check that you have internet connection
- Try closing Terminal and starting over

## What the Application Does

✅ Creates personalized travel itineraries
✅ Suggests activities based on your interests  
✅ Provides day-by-day plans
✅ Includes practical tips and budget estimates
✅ Can save your itinerary to a text file

Enjoy planning your next adventure! 🌍✈️
