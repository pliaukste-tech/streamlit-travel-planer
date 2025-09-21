"""
Travel Planner Application using Google AI Studio

This application helps users plan their travel itinerary by asking for:
- Country to visit
- Activities they like
- Days to spend in the country
- Report weight (long or short)
"""

import os
from google import genai
from google.genai import types

def get_api_key():
    """Get Google AI Studio API key from user input."""
    print("🔑 Google AI Studio API Key Required")
    print("=" * 40)
    print("You can get your API key from: https://aistudio.google.com/app/apikey")
    print("The API key will look like: AIza...")
    print()
    
    while True:
        api_key = input("Please enter your Google AI Studio API key: ").strip()
        
        if not api_key:
            print("❌ API key cannot be empty. Please try again.")
            continue
        
        if not api_key.startswith('AIza'):
            print("⚠️  Warning: API key should start with 'AIza'. Are you sure this is correct?")
            confirm = input("Continue anyway? (y/n): ").strip().lower()
            if confirm not in ['y', 'yes']:
                continue
        
        return api_key

def setup_client(api_key):
    """Initialize the Google AI client with API key."""
    try:
        print("🔄 Connecting to Google AI Studio...")
        client = genai.Client(api_key=api_key)
        
        # Test the connection by trying to list models
        try:
            models = list(client.models.list(config={'page_size': 1}))
            print("✅ Successfully connected to Google AI Studio!")
            return client
        except Exception as e:
            print(f"❌ API key validation failed: {e}")
            print("Please check that your API key is correct and has proper permissions.")
            return None
            
    except Exception as e:
        print(f"❌ Error initializing Google AI client: {e}")
        return None

def get_user_inputs():
    """Collect travel preferences from the user."""
    print("📝 Travel Planning Details")
    print("=" * 40)
    
    # Get country
    country = input("Which country would you like to visit? (e.g., Italy): ").strip()
    while not country:
        country = input("Please enter a valid country name: ").strip()
    
    # Get activities
    activities = input("What activities do you enjoy? (e.g., Hiking, Cooking): ").strip()
    while not activities:
        activities = input("Please enter at least one activity: ").strip()
    
    # Get number of days
    while True:
        try:
            days = int(input("How many days will you spend there? "))
            if days > 0:
                break
            else:
                print("Please enter a positive number of days.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Get report weight
    while True:
        report_weight = input("Would you like a 'long' or 'short' itinerary? ").strip().lower()
        if report_weight in ['long', 'short']:
            break
        else:
            print("Please enter either 'long' or 'short'.")
    
    # Get additional prompt (optional)
    print()
    print("💭 Optional: Ask something specific about your destination")
    print("Examples:")
    print("  - Tell me something important about the history of [country]")
    print("  - What are the local customs I should know about?")
    print("  - What's the best local cuisine to try?")
    additional_prompt = input("Your question (press Enter to skip): ").strip()
    
    return country, activities, days, report_weight, additional_prompt

def create_travel_prompt(country, activities, days, report_weight, additional_prompt=""):
    """Create a detailed prompt for the AI to generate travel itinerary."""
    
    detail_instruction = {
        'short': "Please provide a concise itinerary with key highlights and essential information.",
        'long': "Please provide a detailed itinerary with comprehensive information, including specific locations, timing, tips, and alternative options."
    }
    
    prompt = f"""
You are an expert travel planner. Create a {report_weight} travel itinerary for the following trip:

📍 **Destination:** {country}
🎯 **Interests:** {activities}
📅 **Duration:** {days} days

{detail_instruction[report_weight]}

Please structure the itinerary as follows:
1. **Trip Overview** - Brief introduction to the destination
2. **Daily Itinerary** - Day-by-day breakdown
3. **Activity Recommendations** - Based on the specified interests: {activities}
4. **Practical Tips** - Transportation, accommodation suggestions, local customs
5. **Budget Estimation** - Rough cost estimates
6. **Best Time to Visit** - Weather and seasonal considerations

Make the itinerary practical, engaging, and tailored to someone interested in {activities}.
"""
    
    # Add additional prompt if provided
    if additional_prompt and additional_prompt.strip():
        prompt += f"""

📝 **Additional Information Requested:**
Please also include a dedicated section answering this specific question about {country}:
"{additional_prompt}"

Please provide comprehensive and interesting information for this additional request.
"""
    
    return prompt

def generate_itinerary(client, prompt):
    """Generate travel itinerary using Google AI."""
    try:
        print("\n🤖 Generating your personalized travel itinerary...")
        print("This may take a few moments...\n")
        
        # Configure the generation parameters
        config = types.GenerateContentConfig(
            temperature=0.7,  # Balance creativity and consistency
            top_p=0.9,
            max_output_tokens=2048,
            system_instruction="You are a helpful and knowledgeable travel advisor who creates detailed, practical, and engaging travel itineraries."
        )
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=prompt,
            config=config
        )
        
        return response.text
        
    except Exception as e:
        print(f"❌ Error generating itinerary: {e}")
        return None

def display_itinerary(itinerary):
    """Display the generated itinerary in a formatted way."""
    print("=" * 60)
    print("🎉 YOUR PERSONALIZED TRAVEL ITINERARY")
    print("=" * 60)
    print()
    print(itinerary)
    print()
    print("=" * 60)
    print("✈️ Have a wonderful trip!")

def main():
    """Main application function."""
    print("🌍 Welcome to the Travel Planner!")
    print("=" * 50)
    print()
    
    # Get API key from user
    api_key = get_api_key()
    
    # Setup Google AI client
    client = setup_client(api_key)
    if not client:
        print("\n❌ Unable to connect to Google AI Studio.")
        print("Please check your API key and try again.")
        return
    
    print()
    
    try:
        # Get user inputs
        country, activities, days, report_weight, additional_prompt = get_user_inputs()
        
        # Create prompt for AI
        prompt = create_travel_prompt(country, activities, days, report_weight, additional_prompt)
        
        # Generate itinerary
        itinerary = generate_itinerary(client, prompt)
        
        if itinerary:
            # Display results
            display_itinerary(itinerary)
            
            # Ask if user wants to save the itinerary
            save_choice = input("\nWould you like to save this itinerary to a file? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                filename = f"travel_itinerary_{country.lower().replace(' ', '_')}_{days}days.txt"
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"Travel Itinerary for {country}\n")
                        f.write(f"Activities: {activities}\n")
                        f.write(f"Duration: {days} days\n")
                        f.write(f"Report Type: {report_weight}\n")
                        if additional_prompt and additional_prompt.strip():
                            f.write(f"Additional Question: {additional_prompt.strip()}\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(itinerary)
                        f.write(f"Duration: {days} days\n")
                        f.write(f"Report Type: {report_weight}\n")
                        f.write("=" * 60 + "\n\n")
                        f.write(itinerary)
                    print(f"✅ Itinerary saved to: {filename}")
                except Exception as e:
                    print(f"❌ Error saving file: {e}")
        else:
            print("❌ Failed to generate itinerary. Please try again.")
            
    except KeyboardInterrupt:
        print("\n\n👋 Travel planning cancelled. See you next time!")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()