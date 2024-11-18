import time
import os
import streamlit as st
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from github_scraper import scrape_github_repo

# Access API keys from Streamlit secrets
gemini = st.secrets["GEMINI_API_KEY"]
groq = st.secrets["GROQ_API_KEY"]
github = st.secrets["GITHUB"]


# LLM configuration
llm = ChatGroq(
    model="llama3-70b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=groq
)

# Folder to store downloaded files
DOWNLOAD_FOLDER = "downloaded_files"

# Load documents from a folder path in smaller batches with retries
def load_documents_in_batches(folder_path, batch_size=10, max_retries=3, retry_delay=5):
    attempt = 0
    while attempt < max_retries:
        loader = DirectoryLoader(folder_path, glob="**/*", loader_cls=TextLoader)
        all_documents = loader.load()
        if all_documents:
            batches = [all_documents[i:i + batch_size] for i in range(0, len(all_documents), batch_size)]
            return batches
        attempt += 1
        time.sleep(retry_delay)  # Wait before retrying

    return []

# Split documents into chunks
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    return splitter.split_documents(documents)

# Create the vector store from documents
def create_vector_store(documents):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini)
    return FAISS.from_documents(documents, embedding=embeddings)

# Setup QA chain using vector store
def setup_qa_chain(vector_store):
    retriever = vector_store.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# Streamlit UI
st.set_page_config(layout="wide")  # Set the page layout to wide for a more spacious interface

# Sidebar for repo URL input
with st.sidebar:
    st.title("GitHub Repository Chatbot")
    repo_url = st.text_input("Enter the GitHub repository URL:")

    if st.button("Scrape and Build Chatbot"):
        if repo_url:
            with st.spinner("Scraping repository..."):
                try:
                    message = scrape_github_repo(repo_url, DOWNLOAD_FOLDER, github)
                    st.success(message)
                except Exception as e:
                    st.error(f"Error scraping repository: {e}")

            with st.spinner("Loading and processing documents..."):
                try:
                    # Load documents in batches with retry logic
                    document_batches = load_documents_in_batches(DOWNLOAD_FOLDER, batch_size=10)
                    if not document_batches:
                        st.error("No documents to process.")
                    else:
                        all_chunks = []

                        # Process each batch incrementally
                        for batch in document_batches:
                            chunks = split_documents(batch)
                            all_chunks.extend(chunks)

                        if not all_chunks:
                            st.error("No valid chunks to add to the vector store.")
                        else:
                            vector_store = create_vector_store(all_chunks)
                            if vector_store:
                                qa_chain = setup_qa_chain(vector_store)
                                st.session_state["qa_chain"] = qa_chain
                                st.success("Chatbot is ready!")
                            else:
                                st.error("Failed to create vector store.")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")

# Main chat section in the center of the screen
if "qa_chain" in st.session_state:
    st.title("Ask Questions About the Repository")
    query = st.text_input("Your Question:")
    if st.button("Ask") and query:
        with st.spinner("Fetching answer..."):
            try:
                response = st.session_state["qa_chain"]({"query": query})
                st.write(f"**Answer:** {response['result']}")
                st.write("**Source Files:**", [doc.metadata["source"] for doc in response["source_documents"]])
            except Exception as e:
                st.error(f"Error fetching answer: {e}")
