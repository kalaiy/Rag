import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess

def load_vector_store():
    index = faiss.read_index("vector_store/faiss.index")
    with open("vector_store/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def query_rag(query, model, index, chunks):
    q_embedding = model.encode([query])[0]
    D, I = index.search(np.array([q_embedding]), k=5)
    retrieved = [chunks[i] for i in I[0]]
    context = "\n".join(retrieved)

    prompt = f"""You are an expert on Remetrica internal documentation.
Refer to this context:
{context}

Answer the user's question: {query}
"""

    result = subprocess.check_output(["ollama", "run", "mistral"], input=prompt.encode())
    return result.decode()

st.title("Remetrica RAG Assistant")
query = st.text_input("Ask a question about your DLL or XML API help:")

if query:
    with st.spinner("Thinking..."):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        index, chunks = load_vector_store()
        answer = query_rag(query, model, index, chunks)
        st.markdown("### 📘 Answer:")
        st.write(answer)