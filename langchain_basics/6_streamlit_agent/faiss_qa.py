import streamlit as st
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
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

llm = AzureChatOpenAI(
    openai_api_key=openai_key,
    openai_api_base=openai_api_base,
    deployment_name=deployment_name,
    openai_api_version=openai_api_version,  # Or your specific version
    temperature=0.2,
    max_tokens=1000,
    max_retries=3
)

# 🧠 Embeddings with SentenceTransformer
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 🖼️ Streamlit UI
st.set_page_config(page_title="📄 Local FAISS Search", page_icon="📘")
st.title("📘 Semantic Search with FAISS + SentenceTransformer")

uploaded_file = st.file_uploader("Upload a .txt file", type="txt")
question = st.text_input("Ask something about the document:")
previous_text = ""

if not uploaded_file:
    st.warning("Please upload a .txt file to proceed.")
else:
    # Read file and preview
    text = uploaded_file.read().decode("utf-8")
    if text != previous_text:
        previous_text = text
        st.text_area("📄 File Content", text[:1000], height=200)

        # Split the document into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=40,
            length_function=len,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        documents = [Document(page_content=chunk) for chunk in splitter.split_text(text)]

        # Display information about the split documents
        print("\n--- Document Chunks Information ---")
        print(f"Number of document chunks: {len(documents)}")
        print(f"Sample chunk:\n{documents[0].page_content}\n")
        
        # Vector DB
        vectorstore = FAISS.from_documents(documents, embedding_model)

        # Retriever-based QA
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Adjust k for more/less results
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            verbose=True
        )

    if question:
        print(f"--- Question: {question}")
        with st.spinner("Searching..."):
            result = qa_chain({"query": question})

        st.markdown("### 💬 Answer")
        st.write(result["result"])

        st.markdown("### 🔍 Source Text")
        for i, doc in enumerate(result["source_documents"]):
            st.markdown(f"**Chunk {i+1}:**")
            st.code(doc.page_content.strip())
