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

import subprocess
import sys

def run(cmd):
    print(f"\n{'='*60}")
    print(f"Running: {cmd}")
    print('='*60)
    result = subprocess.run(cmd, shell=True, check=True)
    return result

if __name__ == "__main__":
    run("python src/ingest.py")
    run("python src/generate_synthetic_data.py")
    run("python src/train_classifier.py")
    run("python src/indexer.py")
    print("\n✅ Setup complete! Run: streamlit run app.py")
