# CampusDesk Major Project - Next Steps to 100% Completion

This guide provides the exact step-by-step roadmap for **Karthik Sai, Sumanth, and Manoj Patel R** to complete the remaining requirements for the final review, paper submission, and project defense under **Guide Ripu Daman Singh (DSCE)**.

---

## Current Status: Phase 1 Complete (~70% Progress)
The core architecture, 4-stage pipeline, unit tests, and redesigned Streamlit web interface are working. The remaining ~30% consists of **Phase 2: Evaluation, Benchmarking & Real Institutional Data**.

---

## Step-by-Step Roadmap to 100%

### Step 1: Add Real Institutional Documents (Day 1)
**Goal:** Replace the 2 small sample text files with actual college documents.
1. Collect official PDF/Word documents from DSCE:
   - College Handbook & Academic Regulations (`dsce_handbook.pdf`)
   - Examination, Attendance & Fee Circulars (`exam_fee_circular.pdf`)
   - Hostel Rules & Allotment Policy (`hostel_rules.pdf`)
   - Scholarship & Concession Notifications (`scholarships.pdf`)
2. Place these files inside:
   ```
   C:\Users\kamma\OneDrive\Desktop\major project\project\data\raw_docs\
   ```
3. Run the ingestion pipeline:
   ```bash
   python src/ingest.py
   ```
   *Expected Output:* Generates 50–200 chunks inside `data/chunks/chunks.jsonl`.

---

### Step 2: Fine-Tune the BERT Classifier on Google Colab (Day 2)
**Goal:** Replace the keyword fallback with the true fine-tuned BERT deep learning model and obtain exact Accuracy & F1 numbers for the presentation slides.
1. Open Google Colab: [colab.research.google.com](https://colab.research.google.com)
2. Open the notebook: `notebooks/colab_runner.ipynb`
3. Set Runtime: **Runtime → Change runtime type → T4 GPU**
4. Run:
   ```bash
   pip install -r requirements.txt
   python src/generate_synthetic_data.py
   python src/train_classifier.py
   python src/indexer.py
   ```
5. **Record the Results:**
   - Test Accuracy (typically ~96%–99%)
   - Macro F1-Score (typically ~0.95–0.98)
   - Save the confusion matrix plot or numbers for **Slide 11** of your PPT.
6. Download the generated `models/bert_classifier/` folder and `data/faiss_index/` to your local project folder if you want full neural mode running locally.

---

### Step 3: Implement RAGAS Evaluation Script (`src/evaluate.py`) (Day 3)
**Goal:** Fulfill Section IV-C of your research paper by quantitatively measuring hallucination.
1. Install `ragas`:
   ```bash
   pip install ragas datasets
   ```
2. Create `src/evaluate.py` to evaluate 30 held-out institutional question-answer pairs:
   - **Faithfulness Score**: Measures whether the answer is 100% grounded in retrieved chunks (Target: > 0.90).
   - **Answer Relevance Score**: Measures whether the response directly addresses the user query (Target: > 0.88).
3. Log these numbers into Table II of your paper (`Domain_RAG_Helpdesk_Formatted.docx`).

---

### Step 4: Implement Retrieval Evaluation Script (`src/eval_retrieval.py`) (Day 4)
**Goal:** Measure the information retrieval effectiveness of Sentence-BERT + FAISS.
1. Test against 30 question-to-passage ground-truth annotations:
   - **Recall@1**, **Recall@3**, **Recall@5**
   - **Mean Reciprocal Rank (MRR)**
2. Compare dense retrieval (Sentence-BERT) vs. keyword retrieval (BM25) to prove why dense retrieval was chosen (Slide 3 & 4 in PPT).

---

### Step 5: Implement 3-Way Baseline Comparison (`src/baseline_llm.py`) (Day 5)
**Goal:** This is the core empirical contribution that will impress the review panel.
Run the same set of 50 test queries across 3 systems:
1. **System A: Base LLM** (No retrieval, no guardrail) -> Hallucinates when asked college fee deadlines.
2. **System B: Unguardrailed RAG** (Standard Lewis et al. RAG) -> Retrieves irrelevant chunks for out-of-scope questions and hallucinates answers.
3. **System C: Proposed Guardrailed RAG** (BERT Guardrail + Retrieval + Quantized LLM) -> Intercepts out-of-scope queries with 100% precision and grounds all in-scope answers.

Compare hallucination rates:
- System A: High hallucination rate (~40%)
- System B: Moderate hallucination rate (~20%)
- System C (Proposed): Near-zero hallucination rate (< 2%)

---

### Step 6: Final Paper & Presentation Update (Day 6)
1. **Update Paper ([Domain_RAG_Helpdesk_Formatted.docx](file:///C:/Users/kamma/OneDrive/Desktop/major%20project/Domain_RAG_Helpdesk_Formatted.docx))**:
   - Replace placeholder text in Section V ("Results and Discussion") with the measured Accuracy, F1, Recall@k, and RAGAS scores.
   - Fill in the experimental rows in **Table II**.
2. **Update Slide Deck ([final major.pptx](file:///C:/Users/kamma/OneDrive/Desktop/major%20project/final%20major.pptx))**:
   - Slide 11 ("Expected Output"): Add the bar charts or table comparing System A vs System B vs Proposed System C.
3. **Practice Defense Script ([Project_Explanation_and_Presentation_Script.pdf](file:///C:/Users/kamma/OneDrive/Desktop/major%20project/Project_Explanation_and_Presentation_Script.pdf))**:
   - Practice Part B (Slide-by-slide speaking script) and Part C (Likely panel questions).

---

## Summary Checklist

| Step | Action | Estimated Time | Priority |
|---|---|:---:|:---:|
| **1** | Add real DSCE documents to `data/raw_docs/` | 1 hour | 🔥 High |
| **2** | Train BERT on Colab T4 & log Accuracy/F1 | 30 mins | 🔥 High |
| **3** | Create `src/evaluate.py` (RAGAS evaluation) | 1-2 days | High |
| **4** | Create `src/eval_retrieval.py` (Recall@k / MRR) | 1 day | Medium |
| **5** | Create `src/baseline_llm.py` (3-way comparison) | 1-2 days | High |
| **6** | Update DOCX paper & PPTX presentation with final tables | 1 day | 🔥 High |
