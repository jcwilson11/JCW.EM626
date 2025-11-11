"""
Mini RAG Pipeline for Risk Analysis (Markdown or Text Source)
-----------------------------------------------------------------
This script demonstrates a simple RAG process using local embeddings.
It supports both Markdown (.md) and plain text (.txt) files.

Steps:
1. Load a text or Markdown file
2. Clean Markdown if needed
3. Split text into chunks
4. Generate local embeddings (sentence-transformers)
5. Store embeddings in DuckDB
6. Retrieve most relevant chunks for a query

No API key needed. Works fully offline after installing the model.
"""

import os
import re
import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------- SETTINGS ----------
file_path = "document.md"   # Can also be "whitepaper_excerpt.txt"
db_path = "rag_risk_local.duckdb"
model_name = "sentence-transformers/all-MiniLM-L6-v2"


# ---------- STEP 1: LOAD & CLEAN FILE ----------
def load_text_auto(path: str) -> str:
    """
    Load text from a file with auto encoding detection and basic Markdown cleanup.
    Supports UTF-8 and UTF-16.
    """
    # Try UTF-8, then UTF-16 fallback
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(path, "r", encoding="utf-16") as f:
            text = f.read()

    # If markdown, clean it a bit
    if path.lower().endswith(".md"):
        text = markdown_to_plain_text(text)
    return text


def markdown_to_plain_text(md: str) -> str:
    """Lightweight Markdown cleanup to focus on content."""
    md = re.sub(r"```.*?```", "", md, flags=re.DOTALL)  # remove code blocks
    md = re.sub(r"^#+\s*", "", md, flags=re.MULTILINE)  # remove headers
    md = re.sub(r"`([^`]*)`", r"\1", md)                # inline code
    md = md.replace("**", "").replace("__", "")
    md = md.replace("*", "").replace("_", "")
    md = re.sub(r"\n\s*\n+", "\n\n", md)
    return md


def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50):
    """Split text into overlapping word-based chunks."""
    words = text.split()
    chunks = []
    step = max(1, chunk_size - overlap)

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


# Detect file type
ext = os.path.splitext(file_path)[1].lower()
print(f"Loading file: {file_path} (detected type: {ext})")

text = load_text_auto(file_path)
chunks = chunk_text(text)
print(f"Loaded {len(chunks)} chunks from {file_path}")


# ---------- STEP 2: GENERATE LOCAL EMBEDDINGS ----------
print("Loading embedding model...")
embedder = SentenceTransformer(model_name)

embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
print("Embeddings generated locally.")


# ---------- STEP 3: STORE IN DUCKDB ----------
conn = duckdb.connect(db_path)
conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id INTEGER,
        text TEXT,
        embedding BLOB
    )
""")
conn.execute("DELETE FROM chunks")

for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    conn.execute(
        "INSERT INTO chunks VALUES (?, ?, ?)",
        [i, chunk, emb.astype(np.float32).tobytes()],
    )

conn.commit()
print(f"Stored {len(chunks)} embeddings in {db_path}")


# ---------- STEP 4: QUERY & RETRIEVAL ----------
query_text = input("\nEnter your query (e.g., 'What are the main AI risks?'):\n> ")
query_emb = embedder.encode([query_text], convert_to_numpy=True)[0].astype(np.float32)

data = conn.execute("SELECT id, text, embedding FROM chunks").fetchall()
ids, texts, stored_embeddings = zip(*data)
stored_embeddings = [np.frombuffer(e, dtype=np.float32) for e in stored_embeddings]

similarities = [cosine_similarity([query_emb], [e])[0][0] for e in stored_embeddings]
top_k = np.argsort(similarities)[::-1][:3]

print("\n--- Top Retrieved Chunks ---\n")
for idx in top_k:
    print(f"[Chunk {ids[idx]} | Similarity: {round(similarities[idx], 3)}]")
    print(texts[idx][:600].replace("\n", " "))
    print("\n-------------------------------\n")

# ---------- STEP 5: REFLECTION ----------
print("Interpret the retrieved text and identify relevant risks, likelihoods, and mitigation strategies.")
