
#!/usr/bin/env python3
"""
Text comparison pipeline (Pro vs Con) — Steps 1-2

Step 1: Cleaning  (already implemented)
Step 2: Sentiment (VADER) on CLEANED TEXT using chunk-and-average (Option 2)

Usage examples:
    # Step 1 - cleaning
    python text_compare.py --pros /mnt/data/pros.txt --cons /mnt/data/cons.txt \
        --stopwords /mnt/data/stopwords_en.txt --step clean \
        --domain-stopwords gig economy

    # Step 2 - sentiment on cleaned text (chunk & average)
    python text_compare.py --cleaned-pros /mnt/data/cleaned_pros.txt \
        --cleaned-cons /mnt/data/cleaned_cons.txt \
        --step sentiment --chunk-size 100 --overlap 0
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set, Dict

# --------------- Cleaning Utilities ---------------

URL_PATTERN = re.compile(
    r"(https?://\S+)|"             # http/https URLs
    r"(www\.\S+)|"                 # www. URLs
    r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"  # emails
)

NUMBER_PATTERN = re.compile(r"\b\d+(?:[\.,]\d+)*\b")  # numbers (incl. 1,234 or 12.5)

PUNCT_AND_SYMBOLS = re.compile(r"[^\w\s']+", flags=re.UNICODE)  # keep letters, digits, underscore, whitespace, apostrophe

WHITESPACE = re.compile(r"\s+", flags=re.UNICODE)

TOKEN_PATTERN = re.compile(r"\b[\w']+\b", flags=re.UNICODE)


def load_stopwords(fp: Path) -> Set[str]:
    words: Set[str] = set()
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if not w:
                continue
            words.add(w)
    # Always ignore single and double letters; they get filtered anyway but safe to add them
    words.update({chr(c) for c in range(ord('a'), ord('z')+1)})
    return words


def normalize_text(text: str) -> str:
    # Lowercase
    text = text.lower()
    # Remove URLs, emails
    text = URL_PATTERN.sub(" ", text)
    # Remove numbers
    text = NUMBER_PATTERN.sub(" ", text)
    # Replace punctuation/symbols (keep word chars and apostrophes for contractions)
    text = PUNCT_AND_SYMBOLS.sub(" ", text)
    # Normalize whitespace and strip
    text = WHITESPACE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def filter_tokens(tokens: Iterable[str], stopwords: Set[str], domain_stopwords: Set[str]) -> List[str]:
    out: List[str] = []
    domain = set(w.lower() for w in domain_stopwords)
    for t in tokens:
        tl = t.lower().strip("'")  # drop stray apostrophes
        if len(tl) < 3:
            continue
        if tl in stopwords or tl in domain:
            continue
        # Exclude tokens that are purely digits or underscores (defensive)
        if tl.isdigit() or tl == "_":
            continue
        out.append(tl)
    return out


def clean_file(fp: Path, stopwords: Set[str], domain_stopwords: Set[str]) -> List[str]:
    raw = fp.read_text(encoding="utf-8", errors="ignore")
    norm = normalize_text(raw)
    toks = tokenize(norm)
    toks = filter_tokens(toks, stopwords, domain_stopwords)
    return toks


def write_list(lines: Iterable[str], out_path: Path) -> None:
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_freqs(tokens: Iterable[str], out_path: Path, top_n: int = 0) -> Counter:
    cnt = Counter(tokens)
    # Optional: write full frequency list as CSV
    with out_path.open("w", encoding="utf-8") as f:
        f.write("token,count\n")
        for tok, c in cnt.most_common():
            f.write(f"{tok},{c}\n")
    return cnt


# --------------- Sentiment (Step 2) ---------------
def chunk_tokens(tokens: List[str], chunk_size: int = 100, overlap: int = 0) -> List[List[str]]:
    """Split a token list into chunks for more stable VADER analysis."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    chunks: List[List[str]] = []
    i = 0
    step = chunk_size - overlap
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        if not chunk:
            break
        chunks.append(chunk)
        i += step
    return chunks


def analyze_sentiment_chunks(chunks: List[List[str]]) -> List[Dict]:
    # Prefer vaderSentiment (no NLTK downloads required)
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except Exception as e:
        raise RuntimeError("vaderSentiment library is required. Please install with: pip install vaderSentiment") from e

    sia = SentimentIntensityAnalyzer()
    rows = []
    for idx, toks in enumerate(chunks):
        text = " ".join(toks)
        s = sia.polarity_scores(text)
        rows.append({
            "chunk_id": idx,
            "n_tokens": len(toks),
            "neg": s["neg"],
            "neu": s["neu"],
            "pos": s["pos"],
            "compound": s["compound"],
        })
    return rows


def summarize_sentiment(rows: List[Dict]) -> Dict:
    import math
    if not rows:
        return {"n_chunks": 0, "mean": {}, "weighted": {}}
    # Simple averages
    mean = {
        "neg": sum(r["neg"] for r in rows) / len(rows),
        "neu": sum(r["neu"] for r in rows) / len(rows),
        "pos": sum(r["pos"] for r in rows) / len(rows),
        "compound": sum(r["compound"] for r in rows) / len(rows),
    }
    # Token-weighted averages
    total_tokens = sum(r["n_tokens"] for r in rows)
    weighted = {
        "neg": sum(r["neg"] * r["n_tokens"] for r in rows) / total_tokens,
        "neu": sum(r["neu"] * r["n_tokens"] for r in rows) / total_tokens,
        "pos": sum(r["pos"] * r["n_tokens"] for r in rows) / total_tokens,
        "compound": sum(r["compound"] * r["n_tokens"] for r in rows) / total_tokens,
    }
    return {"n_chunks": len(rows), "mean": mean, "weighted": weighted}


# --------------- CLI ---------------

def main():
    parser = argparse.ArgumentParser(description="Compare Pro/Con texts. Steps: clean, sentiment")
    parser.add_argument("--pros", type=str, help="Path to pros.txt (raw)")
    parser.add_argument("--cons", type=str, help="Path to cons.txt (raw)")
    parser.add_argument("--stopwords", type=str, help="Path to stopwords_en.txt")
    parser.add_argument("--step", type=str, default="clean", choices=["clean", "sentiment"], help="Pipeline step to run")
    parser.add_argument("--domain-stopwords", nargs="*", default=["gig", "economy"], help="Extra common domain terms to remove")
    parser.add_argument("--outdir", type=str, default="", help="Optional output directory")

    # Step 2 specific
    parser.add_argument("--cleaned-pros", type=str, help="Path to cleaned_pros.txt")
    parser.add_argument("--cleaned-cons", type=str, help="Path to cleaned_cons.txt")
    parser.add_argument("--chunk-size", type=int, default=100, help="Tokens per chunk for sentiment")
    parser.add_argument("--overlap", type=int, default=0, help="Token overlap between consecutive chunks")

    args = parser.parse_args()

    # Prepare outdir
    chosen_base = args.outdir or (Path(args.pros).parent.as_posix() if args.pros else (Path(args.cleaned_pros).parent.as_posix() if args.cleaned_pros else "."))
    outdir = Path(chosen_base).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.step == "clean":
        if not (args.pros and args.cons and args.stopwords):
            raise SystemExit("For step=clean, you must provide --pros, --cons, and --stopwords")
        pros_fp = Path(args.pros).expanduser().resolve()
        cons_fp = Path(args.cons).expanduser().resolve()
        stop_fp = Path(args.stopwords).expanduser().resolve()

        stopwords = load_stopwords(stop_fp)
        domain_stop = set(args.domain_stopwords)

        pros_tokens = clean_file(pros_fp, stopwords, domain_stop)
        cons_tokens = clean_file(cons_fp, stopwords, domain_stop)

        # Save cleaned token lists
        (outdir / "cleaned_pros.txt").write_text("\n".join(pros_tokens) + "\n", encoding="utf-8")
        (outdir / "cleaned_cons.txt").write_text("\n".join(cons_tokens) + "\n", encoding="utf-8")

        # Save frequency tables
        def write_freqs(tokens: Iterable[str], out_path: Path):
            from collections import Counter
            cnt = Counter(tokens)
            with out_path.open("w", encoding="utf-8") as f:
                f.write("token,count\n")
                for tok, c in cnt.most_common():
                    f.write(f"{tok},{c}\n")
            return cnt

        pros_freq = write_freqs(pros_tokens, outdir / "word_freq_pros.csv")
        cons_freq = write_freqs(cons_tokens, outdir / "word_freq_cons.csv")

        # Console summaries
        print("=== Cleaning complete ===")
        print(f"Pros tokens: {len(pros_tokens):,} | Unique: {len(set(pros_tokens)):,}")
        print(f"Cons tokens: {len(cons_tokens):,} | Unique: {len(set(cons_tokens)):,}")
        def head(counter, k: int = 20):
            return ", ".join([f"{w}({c})" for w, c in counter.most_common(k)])
        print("\nTop 20 (Pros):")
        print(head(pros_freq, 20))
        print("\nTop 20 (Cons):")
        print(head(cons_freq, 20))

    elif args.step == "sentiment":
        # Read cleaned token lists
        if not (args.cleaned_pros and args.cleaned_cons):
            raise SystemExit("For step=sentiment, you must provide --cleaned-pros and --cleaned-cons")

        def read_tokens(fp: Path) -> List[str]:
            return [line.strip() for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]

        pros_tokens = read_tokens(Path(args.cleaned_pros))
        cons_tokens = read_tokens(Path(args.cleaned_cons))

        # Chunking
        pros_chunks = chunk_tokens(pros_tokens, chunk_size=args.chunk_size, overlap=args.overlap)
        cons_chunks = chunk_tokens(cons_tokens, chunk_size=args.chunk_size, overlap=args.overlap)

        # Sentiment rows
        pros_rows = analyze_sentiment_chunks(pros_chunks)
        cons_rows = analyze_sentiment_chunks(cons_chunks)

        # Write per-chunk CSVs
        import csv
        with (outdir / "sentiment_chunks_pros.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_id","n_tokens","neg","neu","pos","compound"])
            writer.writeheader()
            writer.writerows(pros_rows)
        with (outdir / "sentiment_chunks_cons.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_id","n_tokens","neg","neu","pos","compound"])
            writer.writeheader()
            writer.writerows(cons_rows)

        # Summaries
        pros_summary = summarize_sentiment(pros_rows)
        cons_summary = summarize_sentiment(cons_rows)

        # Write summary JSON
        summary = {
            "params": {"chunk_size": args.chunk_size, "overlap": args.overlap},
            "pros": pros_summary,
            "cons": cons_summary,
        }
        (outdir / "sentiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        # Console output
        def fmt(s): return f"neg={s['neg']:.3f} neu={s['neu']:.3f} pos={s['pos']:.3f} compound={s['compound']:.3f}"
        print("=== Sentiment (cleaned, chunk & average) ===")
        print(f"Chunks — Pros: {pros_summary['n_chunks']} | Cons: {cons_summary['n_chunks']}")
        print("Mean (Pros):     " + fmt(pros_summary["mean"]))
        print("Weighted (Pros): " + fmt(pros_summary["weighted"]))
        print("Mean (Cons):     " + fmt(cons_summary["mean"]))
        print("Weighted (Cons): " + fmt(cons_summary["weighted"]))

if __name__ == "__main__":
    main()
