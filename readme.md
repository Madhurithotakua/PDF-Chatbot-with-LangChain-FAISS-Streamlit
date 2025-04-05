# 📄 PDF Chatbot with LangChain, FAISS & Streamlit

This is a powerful **PDF chatbot application** built using **LangChain**, **FAISS**, and **Streamlit**. You can upload or drag & drop a PDF file, and the app will turn it into an **intelligent chatbot** that can answer questions based on the document's content.

## 🤖 What This App Does

1. **📤 Upload a PDF** – Drag and drop or select a file.
2. **📚 Extract Text** – All content is extracted from the PDF.
3. **✂️ Split into Chunks** – The text is divided into smaller segments to improve accuracy and context handling.
4. **🔎 Generate Embeddings** – Each chunk is converted into vector representations using a HuggingFace transformer model.
5. **🧠 Build a Vector Store (FAISS)** – All embeddings are stored in a FAISS index, enabling fast and intelligent similarity search.
6. **💬 Ask Questions** – Enter your query and get relevant answers from the document in real-time!

---

## 🛠️ Tech Stack

- **LangChain** – Handles embeddings, text splitting, and search logic.
- **FAISS (Facebook AI Similarity Search)** – A fast and efficient vector search library used to find the most relevant chunks.
- **Streamlit** – Creates a responsive and interactive user interface.
- **HuggingFace Transformers** – Used for generating sentence-level embeddings.
- **PyPDF2** – Reads and extracts text from PDF files.

---

## 🔍 What is FAISS?

FAISS (Facebook AI Similarity Search) is a library that allows you to efficiently search through large amounts of vector data. In this project, it's used to store and search the **text embeddings** created from your PDF. When you ask a question, FAISS helps find the **most relevant chunks** of your PDF by comparing the semantic similarity between your query and the stored vectors.

---

## ✅ Requirements

- Python 3.7+
- `langchain`
- `streamlit`
- `faiss-cpu`
- `PyPDF2`
- `sentence-transformers`

### 📦 Install all dependencies:

```bash
pip install langchain streamlit faiss-cpu PyPDF2 sentence-transformers
