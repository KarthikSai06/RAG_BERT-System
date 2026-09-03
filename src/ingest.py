import os
import json
import config

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest_documents():
    print(f"Reading documents from {config.RAW_DOCS_DIR}...")
    all_chunks = []
    chunk_id = 0
    
    for filename in os.listdir(config.RAW_DOCS_DIR):
        file_path = os.path.join(config.RAW_DOCS_DIR, filename)
        if not os.path.isfile(file_path):
            continue
            
        text = ""
        if filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            print(f"Unsupported format for now: {filename}")
            continue
            
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for chunk in chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "source_file": filename,
                "text": chunk
            })
            chunk_id += 1
            
    print(f"Generated {len(all_chunks)} chunks.")
    
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"Chunks saved to {config.CHUNKS_FILE}")

if __name__ == "__main__":
    ingest_documents()
