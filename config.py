"""
config.py
---------
Single source of truth for all hyperparameters, paths, and model identifiers.
Edit values here; every other module imports from here.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "data")
RAW_DOCS_DIR    = os.path.join(DATA_DIR, "raw_docs")
CHUNKS_DIR      = os.path.join(DATA_DIR, "chunks")
CHUNKS_FILE     = os.path.join(CHUNKS_DIR, "chunks.jsonl")
CLASSIFIER_DATA = os.path.join(DATA_DIR, "classifier_data")
TRAIN_CSV       = os.path.join(CLASSIFIER_DATA, "train.csv")
TEST_CSV        = os.path.join(CLASSIFIER_DATA, "test.csv")
FAISS_DIR       = os.path.join(DATA_DIR, "faiss_index")
FAISS_INDEX     = os.path.join(FAISS_DIR, "index.faiss")
FAISS_META      = os.path.join(FAISS_DIR, "index_meta.jsonl")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
BERT_MODEL_DIR  = os.path.join(MODELS_DIR, "bert_classifier")

# ── Ingestion / Chunking ──────────────────────────────────────────────────────
CHUNK_SIZE      = 250    # tokens per chunk
CHUNK_OVERLAP   = 50     # overlap tokens between consecutive chunks (~20%)

# ── Retrieval ─────────────────────────────────────────────────────────────────
EMBED_MODEL     = "all-MiniLM-L6-v2"   # SentenceTransformer model
TOP_K           = 3                    # passages retrieved per query

# ── BERT Classifier ───────────────────────────────────────────────────────────
BERT_BASE_MODEL = "bert-base-uncased"
BERT_EPOCHS     = 5
BERT_LR         = 2e-5
BERT_BATCH_SIZE = 16
BERT_MAX_LEN    = 128
LABEL2ID        = {
    "in-domain-factual":     0,
    "in-domain-procedural":  1,
    "out-of-domain":         2,
}
ID2LABEL        = {v: k for k, v in LABEL2ID.items()}
NUM_LABELS      = len(LABEL2ID)

# ── Generator (LLM) ───────────────────────────────────────────────────────────
LLM_MODEL_ID    = "mistralai/Mistral-7B-Instruct-v0.2"
USE_4BIT        = True       # bitsandbytes 4-bit quantization
USE_GPU         = True       # set False to force CPU (slow but no VRAM needed)
MAX_NEW_TOKENS  = 512

# ── Fallback response ─────────────────────────────────────────────────────────
FALLBACK_MSG = (
    "I'm sorry, that question appears to be outside the scope of the institutional "
    "helpdesk. Please ask about admissions, fees, examinations, hostel, or "
    "scholarship policies."
)

# ── Ensure directories exist ──────────────────────────────────────────────────
for _dir in [RAW_DOCS_DIR, CHUNKS_DIR, CLASSIFIER_DATA, FAISS_DIR, MODELS_DIR, BERT_MODEL_DIR]:
    os.makedirs(_dir, exist_ok=True)
