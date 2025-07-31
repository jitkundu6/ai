from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import AzureChatOpenAI
import os

# Load environment variables from .env
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_version = "2025-01-01-preview"
deployment_name = "gpt-4.1"
# deployment_name = "gpt-4o"

model = AzureChatOpenAI(
    openai_api_key=openai_key,
    openai_api_base=openai_api_base,
    deployment_name=deployment_name,
    openai_api_version=openai_api_version,  # Or your specific version
    temperature=0.0,
    max_tokens=1000,
    max_retries=3
)

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
]

# ---- LangChain OpenAI Chat Model Example ----

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from OpenAI: {result.content}")


# ---- Anthropic Chat Model Example ----

# Create a Anthropic model
# Anthropic models: https://docs.anthropic.com/en/docs/models-overview
model = ChatAnthropic(model="claude-3-opus-20240229")

result = model.invoke(messages)
print(f"Answer from Anthropic: {result.content}")


# ---- Google Chat Model Example ----

# https://console.cloud.google.com/gen-app-builder/engines
# https://ai.google.dev/gemini-api/docs/models/gemini
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

result = model.invoke(messages)
print(f"Answer from Google: {result.content}")
