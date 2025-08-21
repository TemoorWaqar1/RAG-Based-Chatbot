import streamlit as st
from langchain.document_loaders import PyPDFLoader, DocxLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI

# Initialize OpenAI API key
import os
os.environ["OPENAI_API_KEY"] = 'your-openai-api-key'

# Streamlit UI
st.title("RAG Based Web Assistant")

uploaded_file = st.file_uploader("Upload a PDF, DOCX, or TXT file", type=["pdf", "docx", "txt"])

if uploaded_file:
    # Load text based on file type
    if uploaded_file.type == "application/pdf":
        loader = PyPDFLoader(uploaded_file)
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        loader = DocxLoader(uploaded_file)
    elif uploaded_file.type == "text/plain":
        loader = TextLoader(uploaded_file)
    else:
        st.error("Unsupported file type.")
        st.stop()

    # Load and split the document
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Get user query
    query = st.text_input("Ask a question about the document:")
    if query:
        # Retrieve relevant chunks
        relevant_chunks = vector_store.similarity_search(query, k=3)
        # Generate response
        llm = OpenAI()
        response = llm(" ".join([chunk.page_content for chunk in relevant_chunks]))

        st.subheader("Response")
        st.write(response)