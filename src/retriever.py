import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import config
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError:
    faiss = None
    SentenceTransformer = None

class Retriever:
    def __init__(self):
        print("Loading Retriever...")
        self.model = None
        self.index = None
        self.chunks = []
        if SentenceTransformer is not None:
            try:
                self.model = SentenceTransformer(config.EMBED_MODEL)
            except Exception as exc:
                print(f"Embedding model unavailable: {exc}")
        if self.model is not None and faiss is not None:
            try:
                self.index = faiss.read_index(config.FAISS_INDEX)
                with open(config.FAISS_META, "r", encoding="utf-8") as f:
                    self.chunks = [json.loads(line) for line in f]
            except (OSError, RuntimeError, ValueError) as exc:
                print(f"FAISS index unavailable: {exc}")
        if not self.chunks and os.path.exists(config.CHUNKS_FILE):
            with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
                self.chunks = [json.loads(line) for line in f]
        if not self.chunks and os.path.isdir(config.RAW_DOCS_DIR):
            from src.ingest import ingest_documents
            ingest_documents()
            if os.path.exists(config.CHUNKS_FILE):
                with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
                    self.chunks = [json.loads(line) for line in f]
            
    def retrieve(self, query, k=config.TOP_K):
        if not query or not query.strip() or not self.chunks:
            return []
        if self.index is None or self.model is None or faiss is None:
            query_terms = set(query.lower().split())
            scored = []
            for chunk in self.chunks:
                terms = set(chunk["text"].lower().split())
                score = len(query_terms & terms) / max(len(query_terms), 1)
                if score >= config.MIN_RETRIEVAL_SCORE:
                    item = chunk.copy()
                    item["score"] = float(score)
                    scored.append(item)
            return sorted(scored, key=lambda item: item["score"], reverse=True)[:max(0, k)]
            
        q_emb = self.model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(q_emb)
        
        scores, indices = self.index.search(q_emb, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx].copy()  # avoid mutating cached dict
                chunk["score"] = float(scores[0][i])
                results.append(chunk)
                
        return results
