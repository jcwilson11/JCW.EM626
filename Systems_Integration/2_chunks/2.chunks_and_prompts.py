import argparse
from pathlib import Path
from textwrap import dedent

def chunk_text(text: str, max_chars: int, overlap: int):
    """Simple char-based chunking with overlap."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        if end == n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks

EXTRACTION_PROMPT_TEMPLATE = dedent("""
    You are a systems integration risk analyst.
    You read technical manuals and extract a machine-readable graph of components and dependencies.

    Extract components and dependencies from the following system description.

    Return ONLY valid JSON with this exact schema (no commentary, no markdown):

    {{
      "nodes": [
        {{
          "id": "C1",
          "name": "Power Distribution Unit",
          "type": "hardware|software|data|interface|human|other",
          "description": "1-3 sentence description"
        }}
      ],
      "edges": [
        {{
          "source": "C1",
          "target": "C5",
          "relation": "short_verb_phrase",
          "rationale": "1–2 sentence justification"
        }}
      ]
    }}

    Rules:
    - Use short, stable IDs within this chunk (C1, C2, ...).
    - Only create edges where there is a clear integration dependency
      (data, control, power, timing, safety).
    - Prefer specific relation phrases like "sends_telemetry_to",
      "reads_from", "controls", "supplies_power_to".

    TEXT CHUNK START
    ----------------
    {chunk_text}
    ----------------
    TEXT CHUNK END
""").strip()

README_TEMPLATE = dedent("""
    # Chunked system manual

    This folder was created by `02_prepare_chunks_and_prompts.py`.

    For each `chunk_XX_prompt.txt`:

    1. Open the file and copy its full contents into ChatGPT.
    2. Ask ChatGPT to respond with **only** valid JSON (no backticks, no commentary).
    3. Save the response as `chunk_XX_graph.json` in this same folder.

    When all chunks have `chunk_XX_graph.json` files,
    run `03_build_risk_graph.py --chunks-dir THIS_FOLDER --out-json system_graph.json ...`.
""").strip()

def main():
    parser = argparse.ArgumentParser(
        description="Split a Markdown manual into chunks and create ChatGPT prompts."
    )
    parser.add_argument("--markdown", required=True, help="Input .md file")
    parser.add_argument("--out-dir", required=True, help="Output directory for chunks/prompts")
    parser.add_argument("--max-chars", type=int, default=3000,
                        help="Approximate max characters per chunk")
    parser.add_argument("--overlap", type=int, default=400,
                        help="Overlap between consecutive chunks (chars)")
    args = parser.parse_args()

    md_path = Path(args.markdown)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    text = md_path.read_text(encoding="utf-8")
    chunks = chunk_text(text, max_chars=args.max_chars, overlap=args.overlap)

    for i, ch in enumerate(chunks, start=1):
        num = f"{i:02d}"
        chunk_file = out_dir / f"chunk_{num}.txt"
        prompt_file = out_dir / f"chunk_{num}_prompt.txt"

        chunk_file.write_text(ch, encoding="utf-8")
        prompt_text = EXTRACTION_PROMPT_TEMPLATE.format(chunk_text=ch)
        prompt_file.write_text(prompt_text, encoding="utf-8")

        print(f"Wrote chunk {num}: {chunk_file.name}, {prompt_file.name}")

    (out_dir / "README.md").write_text(README_TEMPLATE, encoding="utf-8")
    print(f"Total chunks: {len(chunks)}. Follow README.md for ChatGPT steps.")

if __name__ == "__main__":
    main()
