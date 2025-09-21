# 🎯 SIMPLEST WAY TO RUN THE TRAVEL PLANNER CHATBOT

## For Complete Beginners - Follow These Exact Steps:

### 1. Get Your API Key (One Time Setup)
- Go to: https://aistudio.google.com/app/apikey
- Sign in with Google
- Click "Create API Key"
- Copy the key (starts with "AIza...")

### 2. Open Terminal
- Press `Cmd + Space`
- Type "Terminal"
- Press Enter

### 3. Go to the Project Folder
Copy and paste this line into Terminal:
```bash
cd /Users/virgilijus/Desktop/Projects/streamlit-travel-planer
```

### 4. Start the Chatbot
Copy and paste this line:
```bash
./start_chatbot.sh
```

### 5. Use the Chatbot
- Your web browser will open automatically
- Enter your API key in the sidebar
- Fill out the travel form:
  - Country: Italy
  - Activities: Hiking, Cooking  
  - Days: 3
  - Detail: short or long
  - Additional question (optional): "Tell me something important about the history of Italy"
- Click "Generate My Travel Itinerary"

### 6. Stop the Chatbot
When you're done, go back to Terminal and press `Ctrl + C`

---

## If Something Goes Wrong:

**"Permission denied" error:**
```bash
chmod +x start_chatbot.sh
./start_chatbot.sh
```

**"No such file" error:**
Make sure you're in the right folder:
```bash
ls
```
You should see files like `streamlit_travel_planner.py`

**Package installation issues:**
```bash
pip3 install streamlit google-genai
```

---

**That's it! The chatbot will create amazing travel plans for you! 🎉**
