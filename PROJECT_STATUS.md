PROJECT STATUS
==============
Domain-Specific RAG Helpdesk System
Repository: https://github.com/KarthikSai06/RAG_BERT-System.git
Last Updated: September 2026


IMPLEMENTED (PHASE 1 - COMPLETE)
=================================

1. config.py
   - All hyperparameters, model identifiers, and file paths in one place
   - BERT model: bert-base-uncased
   - Embedding model: all-MiniLM-L6-v2
   - LLM: mistralai/Mistral-7B-Instruct-v0.2
   - Class labels: in-domain-factual, in-domain-procedural, out-of-domain
   - Auto-creates required directories on import

2. src/ingest.py
   - Loads .txt, .pdf (pdfplumber), and .docx (python-docx) files
   - Sliding window chunking: 250 words per chunk, 50-word overlap (~20%)
   - Filters empty and whitespace-only chunks
   - Writes chunks.jsonl with chunk_id, source_file, and text fields

3. src/generate_synthetic_data.py
   - Generates 300 realistic institutional queries
     * 100 in-domain-factual (fee deadlines, attendance rules, hostel curfew)
     * 100 in-domain-procedural (how to apply for waiver, hostel room, etc.)
     * 100 out-of-domain (general knowledge, small talk, unrelated topics)
   - Stratified 80/20 train/test split
   - Writes train.csv and test.csv

4. src/train_classifier.py
   - Fine-tunes bert-base-uncased with a 3-class classification head
   - Uses HuggingFace Trainer with AdamW, LR=2e-5, 5 epochs, batch=16
   - eval_strategy per epoch (new transformers API, not deprecated evaluation_strategy)
   - metric_for_best_model: macro_f1
   - Saves best model weights and tokenizer to models/bert_classifier/
   - Reports accuracy and macro-F1 on test set

5. src/classifier.py
   - QueryClassifier class that loads the fine-tuned BERT model
   - Falls back to base BERT weights if fine-tuned model not found
   - predict(query) -> label string
   - predict_with_confidence(query) -> (label, confidence_score) tuple
   - Runs on GPU if available, else CPU

6. src/indexer.py
   - Encodes all chunks using Sentence-BERT (all-MiniLM-L6-v2)
   - Casts embeddings to float32 for FAISS compatibility
   - L2-normalizes embeddings for cosine similarity via IndexFlatIP
   - Saves FAISS binary index and metadata JSONL file

7. src/retriever.py
   - Retriever class that loads FAISS index and metadata at startup
   - retrieve(query, k) -> list of top-k chunk dicts with score field
   - Uses .copy() to avoid mutating the cached metadata list
   - Bounds-checks FAISS returned indices (handles -1 from underfull index)
   - Float32 cast on query embedding before FAISS search

8. src/generator.py
   - Generator class loading Mistral-7B-Instruct-v0.2 via HuggingFace
   - 4-bit NF4 quantization via BitsAndBytesConfig when GPU is available
   - CPU fallback with float32 when no GPU detected
   - Uses apply_chat_template for correct [INST] token wrapping
   - Context-only instruction: model must answer from retrieved passages only
   - Returns clean stripped response text

9. src/pipeline.py
   - RAGPipeline class orchestrating all 4 stages in sequence
   - Stage 1: BERT classifier
   - Stage 2 (guardrail): out-of-domain queries return fixed fallback immediately
   - Stage 3: FAISS retrieval
   - Stage 4: LLM generation
   - Records per-stage timing in milliseconds
   - init_generator=False mode for lightweight testing without loading the LLM

10. app.py
    - Streamlit chat interface
    - Caches pipeline with @st.cache_resource (loads once per session)
    - Displays label classification per response
    - Shows retrieved source passages with similarity scores in expander
    - Displays per-stage timing below each response
    - Chat history preserved in st.session_state

11. data/raw_docs/
    - sample_handbook.txt: admissions, fees, examinations, hostel rules
    - scholarship_fee_circular.txt: scholarships, fee payment portal, refund policy

12. notebooks/colab_runner.ipynb
    - Install dependencies
    - Run ingest, generate data, train BERT, build FAISS index
    - Launch Streamlit via localtunnel for public URL

13. setup.py
    - Single command to run all pipeline setup steps in order

14. README.md
    - Architecture overview
    - Project structure
    - Step-by-step quick start
    - Tech stack table
    - References

15. requirements.txt
    - All Python dependencies pinned to minimum compatible versions

16. .gitignore
    - Excludes model weights, FAISS index, generated data, venv, IDE files


NOT YET IMPLEMENTED (PHASE 2)
==============================

1. RAGAS Evaluation Script (src/evaluate.py)
   - Requires: ragas library, held-out question-answer pairs
   - Metrics to compute: Faithfulness, Answer Relevance
   - Script should load pipeline, run on test set, print RAGAS report
   - Estimated effort: 1-2 days

2. Retrieval Metrics Script (src/eval_retrieval.py)
   - Requires: annotated question-to-chunk relevance judgements
   - Metrics: Recall@k (k=1,3,5), Mean Reciprocal Rank (MRR)
   - Compare: dense retrieval vs. BM25 keyword baseline
   - Estimated effort: 1 day

3. Baseline Comparison (src/baseline_llm.py)
   - Requires: same test query set
   - Run 3 variants on identical queries:
     a. Plain LLM with no retrieval, no guardrail
     b. RAG without Stage-1 BERT guardrail
     c. Full proposed system (RAG + BERT guardrail)
   - Compare hallucination rate and RAGAS scores across all 3
   - This is the core quantitative contribution of the paper
   - Estimated effort: 2-3 days

4. Real Institutional Documents
   - Replace sample_handbook.txt and scholarship_fee_circular.txt
     with actual college handbooks, FAQ PDFs, and fee circulars
   - Rebuild FAISS index after adding real documents
   - Action required: obtain documents from institution

5. Real Classifier Training Data
   - Expand synthetic 300-query dataset with real student queries
   - Aim for 500-1000 labelled examples for reliable F1 scores
   - Recommended: collect from institution helpdesk logs
   - Action required: manual labelling or helpdesk query log access

6. LoRA Fine-Tuning of Generator (src/finetune_generator.py)
   - Fine-tune Mistral-7B on institution-specific Q&A pairs using LoRA
   - Reduces residual hallucination on edge cases
   - Requires: curated Q&A dataset, Colab Pro or equivalent GPU
   - Estimated effort: 3-5 days

7. Multilingual Support
   - Extend corpus and classifier to handle queries in regional languages
   - Replace all-MiniLM-L6-v2 with a multilingual sentence-embedding model
   - Estimated effort: 3-4 days

8. Confidence Threshold Guardrail
   - Use predict_with_confidence() output (already implemented in classifier.py)
   - If confidence < threshold (e.g., 0.70), route to human-in-the-loop review
     instead of proceeding with low-confidence classification
   - Estimated effort: half a day

9. Production Deployment Configuration
   - Dockerize the Streamlit app and pipeline
   - Add environment variable management (.env file)
   - Add gunicorn or uvicorn server configuration
   - Estimated effort: 1-2 days


HOW TO RUN
==========

Prerequisites:
    pip install -r requirements.txt

Setup (run once in order):
    python src/ingest.py
    python src/generate_synthetic_data.py
    python src/train_classifier.py
    python src/indexer.py

Or run all setup steps at once:
    python setup.py

Launch UI:
    streamlit run app.py

Google Colab:
    Open notebooks/colab_runner.ipynb and run all cells.
    A public URL will be printed via localtunnel.
