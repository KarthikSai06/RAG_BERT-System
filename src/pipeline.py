import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import config
from src.classifier import QueryClassifier
from src.retriever import Retriever
from src.generator import Generator

class RAGPipeline:
    def __init__(self, init_generator=True):
        print("Initializing RAG Pipeline...")
        self.classifier = QueryClassifier()
        self.retriever = Retriever()
        self.generator = Generator() if init_generator else None
        
    def run(self, query):
        result = {
            "query": query,
            "label": None,
            "answer": None,
            "sources": [],
            "is_fallback": False,
            "timings": {}
        }
        
        # Stage 1: Classification
        t0 = time.time()
        label = self.classifier.predict(query)
        t1 = time.time()
        result["label"] = label
        result["timings"]["classification_ms"] = round((t1 - t0) * 1000, 2)
        
        # Stage 4: Guardrail check
        if label == "out-of-domain":
            result["is_fallback"] = True
            result["answer"] = config.FALLBACK_MSG
            return result
            
        # Stage 2: Retrieval
        t2 = time.time()
        sources = self.retriever.retrieve(query)
        t3 = time.time()
        result["sources"] = sources
        result["timings"]["retrieval_ms"] = round((t3 - t2) * 1000, 2)
        
        # Stage 3: Generation
        if self.generator:
            t4 = time.time()
            answer = self.generator.generate(query, sources)
            t5 = time.time()
            result["answer"] = answer
            result["timings"]["generation_ms"] = round((t5 - t4) * 1000, 2)
        else:
            result["answer"] = "Generator not initialized. Retrieved context: " + " | ".join([s['text'] for s in sources])
            
        return result

if __name__ == "__main__":
    # Test pipeline without loading heavy LLM generator
    pipeline = RAGPipeline(init_generator=False)
    
    test_queries = [
        "What is the semester fee deadline?", # In-domain factual
        "How do I apply for a fee waiver?",   # In-domain procedural
        "Who is the Prime Minister of India?" # Out-of-domain
    ]
    
    for q in test_queries:
        print(f"\nQuery: {q}")
        res = pipeline.run(q)
        print(f"Label: {res['label']}")
        print(f"Fallback: {res['is_fallback']}")
        if not res['is_fallback']:
            print(f"Retrieved {len(res['sources'])} sources")
