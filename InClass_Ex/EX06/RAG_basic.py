"""
Mini RAG Pipeline for Risk Analysis
-----------------------------------------------------------------
This script demonstrates a simple RAG process using local embeddings

Steps:
1. Load a text file
2. Split it into text chunks
3. Generate local embeddings (sentence-transformers)
4. Store embeddings in DuckDB
5. Retrieve most relevant chunks for a risk-related query

No API key needed. Works fully offline after installing the model.
"""

import duckdb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------- SETTINGS ----------
txt_path = "whitepaper_excerpt.txt"   # Text file extracted from the AI Risk whitepaper
db_path = "rag_risk_local.duckdb"
model_name = "sentence-transformers/all-MiniLM-L6-v2"

# ---------- STEP 1: LOAD AND CHUNK TEXT ----------
def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def chunk_text(text, chunk_size=400, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

text = load_text(txt_path)
chunks = chunk_text(text)
print("Loaded", len(chunks), "chunks from", txt_path)

# ---------- STEP 2: GENERATE LOCAL EMBEDDINGS ----------
print("Loading embedding model...")
embedder = SentenceTransformer(model_name)

embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
print("Embeddings generated locally.")

# ---------- STEP 3: STORE IN DUCKDB ----------
conn = duckdb.connect(db_path)
conn.execute("CREATE TABLE IF NOT EXISTS chunks (id INTEGER, text TEXT, embedding BLOB)")
conn.execute("DELETE FROM chunks")

for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
    conn.execute("INSERT INTO chunks VALUES (?, ?, ?)", [i, chunk, emb.tobytes()])

conn.commit()
print("Stored", len(chunks), "embeddings in", db_path)

# ---------- STEP 4: QUERY & RETRIEVAL ----------
query_text = input("\nEnter your query (e.g., 'What are the main AI risks?'):\n> ")
query_emb = embedder.encode([query_text])[0]

data = conn.execute("SELECT id, text, embedding FROM chunks").fetchall()
ids, texts, stored_embeddings = zip(*data)
stored_embeddings = [np.frombuffer(e, dtype=np.float32) for e in stored_embeddings]

similarities = [cosine_similarity([query_emb], [e])[0][0] for e in stored_embeddings]
top_k = np.argsort(similarities)[::-1][:3]

print("\n--- Top Retrieved Chunks ---\n")
for idx in top_k:
    print("[Chunk", ids[idx], "| Similarity:", round(similarities[idx], 3), "]")
    print(texts[idx][:600].replace("\n", " "))
    print("\n-------------------------------\n")

# ---------- STEP 5: REFLECTION ----------
print("Interpret the retrieved text and identify relevant risks, likelihoods and mitigation strategies.")