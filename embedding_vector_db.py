import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
import os

# ----------------------------
# Function to extract text from PDF
# ----------------------------
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ----------------------------
# Function to chunk text
# ----------------------------
def chunk_text(text, chunk_size=512, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = splitter.split_text(text)
    return chunks

# ----------------------------
# Function to create FAISS index
# ----------------------------
def create_faiss_index(chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.from_texts(chunks, embeddings)
    return db

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📚 LangChain + FAISS Document Search")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    # Extract and display text
    with st.spinner("Extracting text..."):
        text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text", text[:1000] + "...", height=200)
    
    # Chunk the text
    with st.spinner("Chunking text..."):
        chunks = chunk_text(text)
        st.write(f"✅ Split into {len(chunks)} chunks")
    
    # Create FAISS index
    with st.spinner("Creating FAISS index..."):
        db = create_faiss_index(chunks)
        st.success("FAISS index created!")

    # Search functionality
    query = st.text_input("Enter your query")
    if query:
        with st.spinner("Searching..."):
            results = db.similarity_search_with_score(query, k=5)
            
            st.write("### Top Matches:")
            for i, (doc, score) in enumerate(results):
                st.write(f"**Result {i+1} (Score: {score:.4f})**")
                st.write(doc.page_content)
                st.write("---")
