"""
Streamlit Travel Planner Application using Google AI Studio

This application helps users plan their travel itinerary using a web interface.
"""

import streamlit as st
from google import genai
from google.genai import types
import time

def setup_client(api_key):
    """Initialize the Google AI client with API key."""
    try:
        client = genai.Client(api_key=api_key)
        
        # Test the connection by trying to list models
        try:
            models = list(client.models.list(config={'page_size': 1}))
            return client, None
        except Exception as e:
            return None, f"API key validation failed: {e}"
            
    except Exception as e:
        return None, f"Error initializing Google AI client: {e}"

def create_travel_prompt(country, activities, days, report_weight, additional_prompt=""):
    """Create a detailed prompt for the AI to generate travel itinerary."""
    
    detail_instruction = {
        'short': "Please provide a concise itinerary with key highlights and essential information.",
        'long': "Please provide a detailed itinerary with comprehensive information, including specific locations, timing, tips, and alternative options."
    }
    
    # Base prompt for the itinerary
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
Format the response with proper markdown for better readability.
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
        # Configure the generation parameters
        config = types.GenerateContentConfig(
            temperature=0.7,  # Balance creativity and consistency
            top_p=0.9,
            max_output_tokens=2048,
            system_instruction="You are a helpful and knowledgeable travel advisor who creates detailed, practical, and engaging travel itineraries. Format your responses with clear markdown headers and structure."
        )
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents=prompt,
            config=config
        )
        
        return response.text
        
    except Exception as e:
        return f"Error generating itinerary: {e}"

def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="🌍 Travel Planner",
        page_icon="✈️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .input-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🌍 Travel Planner Chatbot</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2em; color: #666;">Plan your perfect trip with AI assistance!</p>', unsafe_allow_html=True)
    
    # Sidebar for API key input
    with st.sidebar:
        st.header("🔑 API Configuration")
        st.markdown("Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)")
        
        api_key = st.text_input(
            "Google AI Studio API Key",
            type="password",
            placeholder="Enter your API key (starts with AIza...)",
            help="Your API key is used to connect to Google AI Studio. It will not be stored."
        )
        
        if api_key:
            if api_key.startswith('AIza'):
                st.success("✅ API key format looks correct")
            else:
                st.warning("⚠️ API key should start with 'AIza'")
    
    # Main content area
    if not api_key:
        st.info("👈 Please enter your Google AI Studio API key in the sidebar to get started.")
        st.markdown("""
        ### How to get your API key:
        1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
        2. Sign in with your Google account
        3. Click "Create API Key"
        4. Copy the API key and paste it in the sidebar
        """)
        return
    
    # Input form
    st.markdown('<div class="input-section">', unsafe_allow_html=True)
    st.header("📝 Tell me about your trip!")
    
    col1, col2 = st.columns(2)
    
    with col1:
        country = st.text_input(
            "🏛️ Which country would you like to visit?",
            placeholder="e.g., Italy, Japan, France",
            help="Enter the name of the country you want to travel to"
        )
        
        activities = st.text_input(
            "🎯 What activities do you enjoy?",
            placeholder="e.g., Hiking, Cooking, Museums",
            help="List activities you're interested in, separated by commas"
        )
    
    with col2:
        days = st.number_input(
            "📅 How many days will you spend there?",
            min_value=1,
            max_value=30,
            value=3,
            help="Number of days for your trip"
        )
        
        report_weight = st.selectbox(
            "📄 How detailed should the itinerary be?",
            options=["short", "long"],
            index=0,
            help="Short: Key highlights only | Long: Detailed with comprehensive information"
        )
    
    # Additional prompt section
    st.markdown("### 💭 Additional Question (Optional)")
    additional_prompt = st.text_area(
        "❓ Ask something specific about your destination country",
        placeholder="e.g., Tell me something important about the history of Italy\ne.g., What are the local customs I should know about?\ne.g., What's the best local cuisine to try?",
        help="Ask any specific question about the country you're visiting. This will be included as an additional section in your itinerary.",
        height=100
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Generate itinerary button
    if st.button("🚀 Generate My Travel Itinerary", type="primary", use_container_width=True):
        
        # Validate inputs
        if not country or not activities:
            st.error("❌ Please fill in all the required fields (Country and Activities).")
            return
        
        # Show loading message
        with st.spinner("🤖 Creating your personalized travel itinerary... This may take a few moments."):
            
            # Setup client
            client, error = setup_client(api_key)
            
            if error:
                st.error(f"❌ {error}")
                st.info("Please check your API key and try again.")
                return
            
            # Create prompt
            prompt = create_travel_prompt(country, activities, days, report_weight, additional_prompt)
            
            # Generate itinerary
            itinerary = generate_itinerary(client, prompt)
            
        # Display results
        if itinerary and not itinerary.startswith("Error"):
            st.success("✅ Your travel itinerary has been generated!")
            
            # Create tabs for better organization
            tab1, tab2 = st.tabs(["📋 Your Itinerary", "📝 Trip Summary"])
            
            with tab1:
                st.markdown("---")
                st.markdown(itinerary)
                
                # Download button
                download_content = (f"Travel Itinerary for {country}\n" +
                                  f"Activities: {activities}\n" +
                                  f"Duration: {days} days\n" +
                                  f"Report Type: {report_weight}\n")
                
                if additional_prompt and additional_prompt.strip():
                    download_content += f"Additional Question: {additional_prompt.strip()}\n"
                
                download_content += "=" * 60 + "\n\n" + itinerary
                
                st.download_button(
                    label="📥 Download Itinerary",
                    data=download_content,
                    file_name=f"travel_itinerary_{country.lower().replace(' ', '_')}_{days}days.txt",
                    mime="text/plain"
                )
            
            with tab2:
                st.markdown("### 📊 Trip Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("🏛️ Destination", country)
                
                with col2:
                    st.metric("📅 Duration", f"{days} days")
                
                with col3:
                    st.metric("🎯 Activities", len(activities.split(',')))
                
                with col4:
                    st.metric("📄 Detail Level", report_weight.title())
                
                st.markdown("### 🎯 Your Interests")
                for activity in activities.split(','):
                    st.write(f"• {activity.strip()}")
                
                # Show additional prompt if provided
                if additional_prompt and additional_prompt.strip():
                    st.markdown("### ❓ Additional Question Asked")
                    st.info(additional_prompt.strip())
        else:
            st.error(f"❌ {itinerary}")
            st.info("Please try again with a different input or check your API key.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666; font-size: 0.9em;">'
        'Made with ❤️ using Streamlit and Google AI Studio | '
        'Your API key is never stored or shared'
        '</p>',
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
