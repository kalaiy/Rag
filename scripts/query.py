import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import subprocess

def load_data():
    index = faiss.read_index("vector_store/faiss.index")
    with open("vector_store/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def ask_model(context, query):
    prompt = f"""You are a helpful assistant for interpreting Remetrica modeling documentation.

Context:
{context}

Question:
{query}
"""
    result = subprocess.check_output(["ollama", "run", "mistral"], input=prompt.encode())
    return result.decode()

def main():
    query = input("Enter your question: ")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index, chunks = load_data()
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k=5)
    context = "\n".join([chunks[i] for i in I[0]])
    response = ask_model(context, query)
    print("\n" + response)

if __name__ == "__main__":
    main()