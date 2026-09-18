# PROJECT STATUS & ROADMAP
==============================
Domain-Specific RAG Helpdesk System
Repository: https://github.com/SUMANTH1011/RAG_BERT-System.git
Institution: Dayananda Sagar College of Engineering (DSCE)
Authors: K Karthik Sai, G Sumanth, Manoj Patel R | Guide: Ripu Daman Singh
Last Updated: September 18, 2026
Overall Completion: ~65% - 70%

---

## 1. COMPLETED MILESTONES (PHASE 1 - 100% DONE)

### A. Core Architecture & Modules
1. **`config.py`**
   - Central source of truth for hyperparameters, model checkpoints, paths, and labels.
   - Configured models: `bert-base-uncased`, `all-MiniLM-L6-v2`, `mistralai/Mistral-7B-Instruct-v0.2`.
   - Classes: `in-domain-factual` (0), `in-domain-procedural` (1), `out-of-domain` (2).
   - Auto-creates directory tree on import.

2. **`src/ingest.py`**
   - Supports `.txt`, `.pdf` (pdfplumber), and `.docx` (python-docx).
   - Sliding-window chunking: 250 words, 50-word overlap (~20%).
   - Generates `data/chunks/chunks.jsonl` with unique chunk IDs and source tracking.

3. **`src/generate_synthetic_data.py`**
   - Generates 300 balanced institutional queries across all 3 classes.
   - 80/20 stratified split outputting `train.csv` (240 rows) and `test.csv` (60 rows).

4. **`src/train_classifier.py`**
   - Fine-tunes `bert-base-uncased` 3-class sequence classification head using HuggingFace `Trainer`.
   - Optimized with AdamW, learning rate 2e-5, 5 epochs, batch size 16.
   - Evaluates per epoch targeting `macro_f1`. Saves best weights to `models/bert_classifier/`.

5. **`src/classifier.py`**
   - `QueryClassifier` inference wrapper with automatic fallback heuristic if BERT weights or GPU are unavailable locally.
   - Provides both `predict(query)` and `predict_with_confidence(query)`.

6. **`src/indexer.py`**
   - Sentence-BERT (`all-MiniLM-L6-v2`) embeddings with float32 casting.
   - L2-normalized `IndexFlatIP` FAISS index for exact cosine similarity search.
   - Writes `index.faiss` and `index_meta.jsonl`.

7. **`src/retriever.py`**
   - Loads FAISS index and metadata cache.
   - Performs top-$k$ dense retrieval with fallback to token-overlap lexical scoring on CPU.
   - Immutable chunk copies prevent dictionary mutation across queries.

8. **`src/generator.py`**
   - `Mistral-7B-Instruct-v0.2` generation with 4-bit NF4 `bitsandbytes` quantization on GPU.
   - Context-only prompt wrapping with chat template `[INST]` formatting to eliminate hallucinations.
   - Fallback extractive answering mode for CPU-only systems.

9. **`src/pipeline.py`**
   - 4-stage orchestrator: Classification -> Guardrail Check -> Retrieval -> Grounded Generation.
   - Records granular latency telemetry per stage (`classification_ms`, `retrieval_ms`, `generation_ms`).

### B. User Interface & Integration
10. **`app.py` (Redesigned & Professionalized)**
    - Modern, standard academic/enterprise palette (light slate `#f8fafc` background, crisp cards `#ffffff`, slate borders `#e2e8f0`).
    - Fixed chat input box docking at bottom of screen.
    - Added 1-click sample query buttons in the left sidebar for instant demonstration.
    - Real-time classification badges: `Factual Policy`, `Procedural Guide`, `Out-of-Scope (Guardrail Triggered)`.
    - Expandable drawer with source citations and similarity scores.

### C. Testing & Verification
11. **Unit Test Suite (`tests/`)**
    - 24 automated test cases implemented; **20 passed, 4 skipped** (FAISS index presence check).
12. **`TEST_RUN_RESULTS.md`**
    - Verified all 3 query types (Factual, Procedural, Guardrail Intercept) with full telemetry and zero hallucination.

---

## 2. PENDING WORK (PHASE 2 - NEXT STEPS TO 100% COMPLETION)

### Priority 1: Real Institutional Documents
- [ ] Replace `sample_handbook.txt` and `scholarship_fee_circular.txt` in `data/raw_docs/` with official DSCE documents:
  - College Academic Regulations & Handbook (PDF)
  - Examination & Attendance Circulars (PDF)
  - Hostel Rules & Fee Circulars (PDF/DOCX)
  - Scholarship & Fee Concession Notifications (PDF)
- [ ] Re-run ingestion: `python src/ingest.py`

### Priority 2: BERT Fine-Tuning & Metric Logging on Google Colab
- [ ] Run `notebooks/colab_runner.ipynb` on Google Colab with T4 GPU:
  ```bash
  pip install -r requirements.txt
  python src/train_classifier.py
  ```
- [ ] Record the final evaluation output:
  - Test Accuracy
  - Macro F1-Score
  - Per-class precision & recall
- [ ] Export confusion matrix for the final presentation deck.

### Priority 3: RAGAS Automated Evaluation Script (`src/evaluate.py`)
- [ ] Implement `src/evaluate.py` using `ragas` library.
- [ ] Measure quantitative hallucination metrics:
  - **Faithfulness**: Are generated claims directly grounded in retrieved passages?
  - **Answer Relevance**: Does the generated answer address the question?
- [ ] Generate evaluation summary table for research paper Section IV-C.

### Priority 4: Retrieval Benchmark Script (`src/eval_retrieval.py`)
- [ ] Implement `src/eval_retrieval.py` against 50 held-out query-passage pairs.
- [ ] Compute standard IR metrics:
  - **Recall@1**, **Recall@3**, **Recall@5**
  - **Mean Reciprocal Rank (MRR)**
- [ ] Benchmark dense retrieval (Sentence-BERT) vs. BM25 keyword baseline.

### Priority 5: 3-Way Baseline Comparison (`src/baseline_llm.py`)
- [ ] Implement comparison script running the same 50 test queries across 3 systems:
  1. *Plain LLM* (No retrieval, no guardrail) -> Measure hallucination rate.
  2. *Unguardrailed RAG* (Retrieval + LLM, no Stage-1 BERT) -> Measures hallucination on out-of-scope queries.
  3. *Proposed System* (BERT Guardrail + Dense Retrieval + Quantized LLM) -> Proves zero hallucination on OOD.
- [ ] This provides the empirical proof for Table II in the paper.

### Priority 6: Final Paper & Presentation Update
- [ ] Update Table II in `Domain_RAG_Helpdesk_Formatted.docx` with measured numbers.
- [ ] Update Slide 11 in `final major.pptx` with actual graphs/tables.
- [ ] Prepare final demo recording.

---

## 3. HOW TO RUN QUICK REFERENCE

```bash
# 1. Pipeline Setup (Ingest -> Generate -> Train -> Index)
python setup.py

# 2. Run Test Suite
pytest tests

# 3. Launch Web Interface
streamlit run app.py

# 4. Colab GPU Execution
Open notebooks/colab_runner.ipynb on Colab (Runtime: T4 GPU)
```
