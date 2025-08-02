from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX, _mysql_prompt
from langchain.prompts.prompt import PromptTemplate

import streamlit as st
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import os


# Load environment variables from .env
load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_version = "2025-01-01-preview"
deployment_name = "gpt-4.1"
# deployment_name = "gpt-4o"

# Initialize the LLM with Azure OpenAI
# This will be used to generate SQL queries based on user questions.
# The model is configured with specific parameters like temperature, max tokens, and retries.
# You can adjust these parameters based on your requirements.
# The `AzureChatOpenAI` class is used to connect to the Azure OpenAI service.
# Make sure to replace the deployment name with your actual deployment name in Azure.
# The `openai_api_version` is set to a specific version, which you can change based on your needs.
# The `temperature` parameter controls the randomness of the model's responses.
# A lower temperature (like 0.2) makes the model more deterministic, while a higher value (like 0.9) makes it more creative.
llm = AzureChatOpenAI(
    openai_api_key=openai_key,
    openai_api_base=openai_api_base,
    deployment_name=deployment_name,
    openai_api_version=openai_api_version,  # Or your specific version
    temperature=0.2,
    max_tokens=1000,
    max_retries=3
)
# llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)


# ------------------ Define Few-Shot Examples -------------------
# These examples will be used to guide the model in generating SQL queries based on user questions.
# You can modify these examples based on your database schema and expected queries.
# The examples should be representative of the types of questions users might ask.
# The `few_shots` variable contains a list of dictionaries, each representing a question, SQL query, SQL result, and the expected answer.
# The model will use these examples to learn how to generate SQL queries and interpret their results.   
few_shots = [
    {'Question' : "How many t-shirts do we have left for Nike in XS size and white color?",
     'SQLQuery' : "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Nike' AND color = 'White' AND size = 'XS'",
     'SQLResult': "Result of the SQL query",
     'Answer' : "91"},
    {'Question': "How much is the total price of the inventory for all S-size t-shirts?",
     'SQLQuery':"SELECT SUM(price*stock_quantity) FROM t_shirts WHERE size = 'S'",
     'SQLResult': "Result of the SQL query",
     'Answer': "22292"},
    {'Question': "If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue  our store will generate (post discounts)?" ,
     'SQLQuery' : """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Levi'
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
     'SQLResult': "Result of the SQL query",
     'Answer': "16725.4"} ,
     {'Question' : "If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?" ,
      'SQLQuery': "SELECT SUM(price * stock_quantity) FROM t_shirts WHERE brand = 'Levi'",
      'SQLResult': "Result of the SQL query",
      'Answer' : "17462"},
    {'Question': "How many white color Levi's shirt I have?",
     'SQLQuery' : "SELECT sum(stock_quantity) FROM t_shirts WHERE brand = 'Levi' AND color = 'White'",
     'SQLResult': "Result of the SQL query",
     'Answer' : "290"
     },
    {'Question': "how much sales amount will be generated if we sell all large size t shirts today in nike brand after discounts?",
     'SQLQuery' : """SELECT sum(a.total_amount * ((100-COALESCE(discounts.pct_discount,0))/100)) as total_revenue from
(select sum(price*stock_quantity) as total_amount, t_shirt_id from t_shirts where brand = 'Nike' and size="L"
group by t_shirt_id) a left join discounts on a.t_shirt_id = discounts.t_shirt_id
 """,
     'SQLResult': "Result of the SQL query",
     'Answer' : "290"
    }
]


db_path = "/home/skundu/Documents/MY_PROJECT/ai/langchain_basics/6_streamlit_agent/tshirt_store.db"

def get_few_shot_db_chain():
    # db_user = "root"
    # db_password = "root"
    # db_host = "localhost"
    # db_name = "tshirts"
    # db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}", sample_rows_in_table_info=3)
    
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    to_vectorize = [" ".join(example.values()) for example in few_shots]
    vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=few_shots)
    example_selector = SemanticSimilarityExampleSelector(
        vectorstore=vectorstore,
        k=2,
    )
    mysql_prompt = """You are a MySQL expert. Given an input question, first create a syntactically correct MySQL query to run, then look at the results of the query and return the answer to the input question.
    Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per MySQL. You can order the results to return the most informative data in the database.
    Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in backticks (`) to denote them as delimited identifiers.
    Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
    Pay attention to use CURDATE() function to get the current date, if the question involves "today".
    
    Use the following format:
    
    Question: Question here
    SQLQuery: Query to run with no pre-amble
    SQLResult: Result of the SQLQuery
    Answer: Final answer here
    
    No pre-amble.
    """

    example_prompt = PromptTemplate(
        input_variables=["Question", "SQLQuery", "SQLResult","Answer",],
        template="\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}",
    )

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=mysql_prompt,
        suffix=PROMPT_SUFFIX,
        input_variables=["input", "table_info", "top_k"], #These variables are used in the prefix and suffix
    )
    chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, prompt=few_shot_prompt)
    return chain


# ------------------ Streamlit UI -------------------  🛢 🛢️ ⛁ ⛃
st.set_page_config(page_title="Smart Agentic AI DB", page_icon="🛢️")
st.title("T Shirts: Database Q&A 👕")
question = st.text_input("Question: ")

if question:
    chain = get_few_shot_db_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)
    st.header("DB Context")
    st.write(chain.database.get_context())
else:
    st.warning("Please enter a question to get started.")
    st.info("Example questions:\n"
            "- How many t-shirts do we have left for Nike in XS size and white color?\n"
            "- How much is the total price of the inventory for all S-size t-shirts?\n"
            "- If we have to sell all the Levi’s T-shirts today with discounts applied. How much revenue our store will generate (post discounts)?"
            "- If we have to sell all the Levi’s T-shirts today. How much revenue our store will generate without discount?\n"
            "- How many white color Levi's shirt I have?\n"
            "- How much sales amount will be generated if we sell all large size t shirts today in nike brand after discounts?"
            )
# streamlit run langchain_basics/6_streamlit_agent/sqldb_bot.py
