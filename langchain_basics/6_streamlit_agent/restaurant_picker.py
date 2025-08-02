import os
import streamlit as st
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool
from langchain_community.utilities import SerpAPIWrapper
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv


# Load environment variables from .env
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_version = "2025-01-01-preview"
deployment_name = "gpt-4.1"
# deployment_name = "gpt-4o"

llm = AzureChatOpenAI(
    openai_api_key=openai_key,
    openai_api_base=openai_api_base,
    deployment_name=deployment_name,
    openai_api_version=openai_api_version,  # Or your specific version
    temperature=0.2,
    max_tokens=1000,
    max_retries=3
)


# Sidebar for location and cuisine
st.set_page_config(page_title="🍴 Restaurant Finder", page_icon="🍽️")
st.sidebar.title("🌍 Location & Cuisine")
location = st.sidebar.selectbox("Select Location", [
    "Bangalore", "Mumbai", "Delhi", "Chennai", "Kolkata", "Hyderabad", "Pune"
])

cuisine = st.sidebar.selectbox("Select Cuisine", [
    "Italian", "Chinese", "South Indian", "North Indian", "Japanese", "Mexican", "Thai", "Continental"
])

dish = st.sidebar.selectbox("Select Dish", [
    "Pasta", "Sushi", "Dosa", "Biryani", "Ramen", "Tacos", "Pad Thai", "Pizza"
])

# Search agent setup
search = SerpAPIWrapper()

def search_restaurants(_):
    query = f"Top {cuisine} restaurants in {location} with address and rating for {dish}"
    return search.run(query)

tool = Tool(
    name="RestaurantSearch",
    func=search_restaurants,
    description="Useful for finding restaurants based on location and cuisine"
)

agent = initialize_agent(
    tools=[tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 🔎 Trigger Search
st.title("🍽️ Restaurant & Cuisine Finder")
st.write(f"Looking for **{cuisine}** food / **{dish}** in **{location}**? Let me find the best spots for you! 🚀")

# Streamlit app to find restaurants based on user input
if st.button("Find Restaurants"):
    with st.spinner("Searching tasty places..."):
        try:
            response = agent.run(f"Find restaurants, Looking for {cuisine} food in {location} for {dish}.")
            st.markdown("### 🍜 Recommended Restaurants")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
