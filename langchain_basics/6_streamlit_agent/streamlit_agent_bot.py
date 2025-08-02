import os
import streamlit as st
import wikipedia
from langchain.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
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
    temperature=0.0,
    max_tokens=1000,
    max_retries=3
)

# ------------------ Define Tools -------------------
def wiki_search(query: str) -> str:
    try:
        return wikipedia.summary(query, sentences=3)
    except Exception as e:
        return f"Wiki error: {e}"

search = SerpAPIWrapper()

tools = [
    Tool(
        name="Wikipedia",
        func=wiki_search,
        description="Use for factual or encyclopedic knowledge."
    ),
    Tool(
        name="GoogleSearch",
        func=search.run,
        description="Use for real-time or current events, or if Wikipedia fails."
    )
]

# ------------------ Initialize Agent -------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ------------------ Streamlit UI -------------------
st.set_page_config(page_title="Smart Agentic AI", page_icon="🧠")
st.title("Smart Info-Bot")
st.markdown("Ask anything — the agent uses **Wikipedia** or **Google Search** intelligently.")

user_input = st.text_input("🔎 Enter your question")

# streamlit run script.py
if user_input:
    with st.spinner(" 🤔 Thinking..."):
        try:
            response = agent.run(user_input)
            st.success(response)
        except Exception as e:
            st.error(f"Error: {e}")
