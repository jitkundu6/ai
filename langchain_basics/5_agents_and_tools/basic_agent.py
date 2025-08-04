from langchain.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, AgentExecutor, create_react_agent, AgentType, create_structured_chat_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import Tool, StructuredTool
from langchain import hub
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv
import os


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

# Define Tools  
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime

    now = datetime.datetime.now()
    return now.strftime("%I:%M %p") 

def search_wikipedia(query):
    """Searches Wikipedia and returns the summary of the first result."""
    from wikipedia import summary

    try:
        return summary(query, sentences=2)
    except:
        return "I couldn't find any information on that."
    
def search_google(query):
    """Searches Google and returns the top result."""
    search = SerpAPIWrapper()
    return search.run(query)    


# Define the tools that the agent can use
tools = [
    Tool(
        name="Current Time",
        func=get_current_time,
        description="Returns the current time in H:MM AM/PM format."
    ),
    Tool(
        name="Search Wikipedia",
        func=search_wikipedia,
        description="Searches Wikipedia and returns a summary of the first result."
    ),
    Tool(
        name="Search Google",
        func=search_google,
        description="Searches Google and returns the top result."
    )
]

# Initialize the agent with the tools
# prompt = hub.pull("hwchase17/react")
prompt = hub.pull("hwchase17/structured-chat-agent")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# agent = create_react_agent(
agent = create_structured_chat_agent(
    tools=tools,
    llm=llm,
    prompt=prompt
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    memory=memory,  # Use the conversation memory to maintain context
    verbose=True  # Enable verbose output for debugging
)

# Run the agent with a test query  
def run_agent():
    query = input("You: ")
    if query.lower() == "exit":
        return
    memory.chat_memory.add_messages([HumanMessage(content=query)])
    response = agent_executor.invoke({"input": query})
    print(f"Agent: {response['output']}")
    memory.chat_memory.add_messages([AIMessage(content=response['output'])])
    run_agent()


initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: 'Current Time', 'Search Wikipedia', 'Search Google'."
memory.chat_memory.add_messages([SystemMessage(content=initial_message)])
print("Type 'exit' to quit.")
run_agent()
