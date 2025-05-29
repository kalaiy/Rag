import os
import pickle
import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

def parse_docs(doc_folder):
    texts = []
    for fname in os.listdir(doc_folder):
        fpath = os.path.join(doc_folder, fname)
        if fname.endswith('.html'):
            with open(fpath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'html.parser')
                texts.append(soup.get_text())
        elif fname.endswith('.xml'):
            with open(fpath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f, 'xml')
                texts.append(soup.get_text())
    return texts

def chunk_text(text, max_length=300):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def build_index(chunks, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(embeddings)
    return index, embeddings

def main():
    docs = parse_docs("docs")
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    index, embeddings = build_index(chunks)

    os.makedirs("vector_store", exist_ok=True)
    faiss.write_index(index, "vector_store/faiss.index")
    with open("vector_store/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

if __name__ == "__main__":
    main()