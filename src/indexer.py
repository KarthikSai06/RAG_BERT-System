import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import faiss
import numpy as np
import config
from sentence_transformers import SentenceTransformer

def build_index():
    print(f"Loading embedding model: {config.EMBED_MODEL}")
    model = SentenceTransformer(config.EMBED_MODEL)
    
    chunks = []
    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            
    print(f"Loaded {len(chunks)} chunks.")
    texts = [chunk["text"] for chunk in chunks]
    
    print("Generating embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)
    
    print(f"Saving index to {config.FAISS_INDEX}")
    faiss.write_index(index, config.FAISS_INDEX)
    
    print(f"Saving metadata to {config.FAISS_META}")
    with open(config.FAISS_META, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
            
    print("Indexing complete.")

if __name__ == "__main__":
    build_index()
