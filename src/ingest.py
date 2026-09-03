import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import config

try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from docx import Document as DocxDocument
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False

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
        elif filename.endswith(".pdf"):
            if not PDF_SUPPORT:
                print("pdfplumber not installed. Skipping:", filename)
                continue
            with pdfplumber.open(file_path) as pdf:
                text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif filename.endswith(".docx"):
            if not DOCX_SUPPORT:
                print("python-docx not installed. Skipping:", filename)
                continue
            doc = DocxDocument(file_path)
            text = "\n".join(para.text for para in doc.paragraphs if para.text.strip())
        else:
            print(f"Unsupported format, skipping: {filename}")
            continue
            
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for chunk in chunks:
            if chunk.strip():  # skip empty/whitespace chunks
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "source_file": filename,
                    "text": chunk.strip()
                })
                chunk_id += 1
            
    print(f"Generated {len(all_chunks)} chunks.")
    
    with open(config.CHUNKS_FILE, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"Chunks saved to {config.CHUNKS_FILE}")

if __name__ == "__main__":
    ingest_documents()
