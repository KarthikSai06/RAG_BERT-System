"""
setup.py
--------
Run this script once to set up the full pipeline:
  1. Ingest documents → chunks
  2. Generate synthetic classifier training data
  3. Train BERT classifier
  4. Build FAISS index

Usage:
    python setup.py
"""

import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def run(script):
    print(f"\n{'='*60}")
    print(f"Running: {script}")
    print('='*60)
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / script)],
        cwd=str(ROOT),
        check=True,
    )
    return result

if __name__ == "__main__":
    for script in ("ingest.py", "generate_synthetic_data.py", "train_classifier.py", "indexer.py"):
        run(script)
    print("\nSetup complete! Run: streamlit run app.py")
