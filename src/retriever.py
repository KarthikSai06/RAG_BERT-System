import json
import faiss
import numpy as np
import config
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self):
        print("Loading Retriever...")
        self.model = SentenceTransformer(config.EMBED_MODEL)
        try:
            self.index = faiss.read_index(config.FAISS_INDEX)
            self.chunks = []
            with open(config.FAISS_META, "r", encoding="utf-8") as f:
                for line in f:
                    self.chunks.append(json.loads(line))
        except Exception as e:
            print("FAISS index or metadata not found. Please run indexer.py first.")
            self.index = None
            self.chunks = []
            
    def retrieve(self, query, k=config.TOP_K):
        if self.index is None:
            return []
            
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        
        scores, indices = self.index.search(q_emb, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunk["score"] = float(scores[0][i])
                results.append(chunk)
                
        return results
