
#!/usr/bin/env python3
"""
Text comparison pipeline (Pro vs Con) — Steps 1-3

Step 1: Cleaning
Step 2: Sentiment (VADER) on cleaned text with chunking/averaging
Step 3: Bigrams (frequency + PMI + t-score + Dice)

This file supports running single steps with --step.
"""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set, Dict, Tuple

# --------------- Cleaning Utilities (Step 1) ---------------

URL_PATTERN = re.compile(
    r"(https?://\S+)|"             # http/https URLs
    r"(www\.\S+)|"                 # www. URLs
    r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"  # emails
)

NUMBER_PATTERN = re.compile(r"\b\d+(?:[\.,]\d+)*\b")  # numbers (incl. 1,234 or 12.5)

PUNCT_AND_SYMBOLS = re.compile(r"[^\w\s']+", flags=re.UNICODE)  # keep letters, digits, whitespace, apostrophe

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
    words.update({chr(c) for c in range(ord('a'), ord('z')+1)})
    return words


def normalize_text(text: str) -> str:
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = NUMBER_PATTERN.sub(" ", text)
    text = PUNCT_AND_SYMBOLS.sub(" ", text)
    text = WHITESPACE.sub(" ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text)


def filter_tokens(tokens: Iterable[str], stopwords: Set[str], domain_stopwords: Set[str]) -> List[str]:
    out: List[str] = []
    domain = set(w.lower() for w in domain_stopwords)
    for t in tokens:
        tl = t.lower().strip("'")
        if len(tl) < 3:
            continue
        if tl in stopwords or tl in domain:
            continue
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


# --------------- Sentiment (Step 2) ---------------

def chunk_tokens(tokens: List[str], chunk_size: int = 100, overlap: int = 0) -> List[List[str]]:
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
    if not rows:
        return {"n_chunks": 0, "mean": {}, "weighted": {}}
    mean = {
        "neg": sum(r["neg"] for r in rows) / len(rows),
        "neu": sum(r["neu"] for r in rows) / len(rows),
        "pos": sum(r["pos"] for r in rows) / len(rows),
        "compound": sum(r["compound"] for r in rows) / len(rows),
    }
    total_tokens = sum(r["n_tokens"] for r in rows)
    weighted = {
        "neg": sum(r["neg"] * r["n_tokens"] for r in rows) / total_tokens,
        "neu": sum(r["neu"] * r["n_tokens"] for r in rows) / total_tokens,
        "pos": sum(r["pos"] * r["n_tokens"] for r in rows) / total_tokens,
        "compound": sum(r["compound"] * r["n_tokens"] for r in rows) / total_tokens,
    }
    return {"n_chunks": len(rows), "mean": mean, "weighted": weighted}


# --------------- Bigrams (Step 3) ---------------

def read_token_lines(fp: Path) -> List[str]:
    return [line.strip() for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


def bigram_counts(tokens: List[str]) -> Tuple[Counter, Counter]:
    uni = Counter(tokens)
    bi = Counter()
    for i in range(len(tokens) - 1):
        bi[(tokens[i], tokens[i+1])] += 1
    return uni, bi


def bigram_stats(tokens: List[str], min_count: int = 2) -> List[Dict]:
    """Compute bigram stats: frequency, PMI, t-score, Dice.
    min_count filters out very rare bigrams (default >=2 occurrences).
    """
    uni, bi = bigram_counts(tokens)
    N = len(tokens)
    B = max(1, N - 1)

    rows: List[Dict] = []
    for (w1, w2), c12 in bi.items():
        if c12 < min_count:
            continue
        c1 = uni[w1]
        c2 = uni[w2]
        # PMI
        pmi = math.log2((c12 * N) / (c1 * c2)) if c1 > 0 and c2 > 0 and c12 > 0 else float("-inf")
        # Expected count under independence for t-score
        expected = (c1 * c2) / B
        tscore = (c12 - expected) / math.sqrt(c12) if c12 > 0 else 0.0
        # Dice coefficient
        dice = (2 * c12) / (c1 + c2)
        rows.append({
            "w1": w1, "w2": w2,
            "count": c12,
            "left_count": c1, "right_count": c2,
            "pmi": pmi,
            "t_score": tscore,
            "dice": dice
        })

    # Sort primarily by count desc, then PMI desc
    rows.sort(key=lambda r: (r["count"], r["pmi"]), reverse=True)
    return rows


def write_bigram_csv(rows: List[Dict], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["w1","w2","count","left_count","right_count","pmi","t_score","dice"])
        writer.writeheader()
        writer.writerows(rows)


# --------------- CLI ---------------

def main():
    parser = argparse.ArgumentParser(description="Compare Pro/Con texts. Steps: clean, sentiment, bigrams")
    parser.add_argument("--step", type=str, required=True, choices=["clean","sentiment","bigrams"])

    # Step 1 (clean)
    parser.add_argument("--pros", type=str, help="Path to pros.txt (raw)")
    parser.add_argument("--cons", type=str, help="Path to cons.txt (raw)")
    parser.add_argument("--stopwords", type=str, help="Path to stopwords_en.txt")
    parser.add_argument("--domain-stopwords", nargs="*", default=["gig","economy"])
    parser.add_argument("--outdir", type=str, default="")

    # Step 2 (sentiment)
    parser.add_argument("--cleaned-pros", type=str, help="Path to cleaned_pros.txt")
    parser.add_argument("--cleaned-cons", type=str, help="Path to cleaned_cons.txt")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=0)

    # Step 3 (bigrams)
    parser.add_argument("--min-count", type=int, default=2, help="Minimum bigram frequency to include")

    args = parser.parse_args()

    # Resolve outdir
    def resolve_outdir():
        candidates = [args.outdir, args.pros, args.cleaned_pros, "."]
        for c in candidates:
            if c:
                p = Path(c).expanduser()
                return p.parent.resolve() if p.is_file() else p.resolve()
        return Path(".").resolve()

    outdir = resolve_outdir()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.step == "clean":
        if not (args.pros and args.cons and args.stopwords):
            raise SystemExit("For step=clean, you must provide --pros, --cons, and --stopwords")
        stopwords = load_stopwords(Path(args.stopwords))
        domain_stop = set(args.domain_stopwords)
        pros_tokens = clean_file(Path(args.pros), stopwords, domain_stop)
        cons_tokens = clean_file(Path(args.cons), stopwords, domain_stop)

        (outdir / "cleaned_pros.txt").write_text("\n".join(pros_tokens) + "\n", encoding="utf-8")
        (outdir / "cleaned_cons.txt").write_text("\n".join(cons_tokens) + "\n", encoding="utf-8")

        def write_freqs(tokens: Iterable[str], out_path: Path):
            cnt = Counter(tokens)
            with out_path.open("w", encoding="utf-8") as f:
                f.write("token,count\n")
                for tok, c in cnt.most_common():
                    f.write(f"{tok},{c}\n")
            return cnt

        write_freqs(pros_tokens, outdir / "word_freq_pros.csv")
        write_freqs(cons_tokens, outdir / "word_freq_cons.csv")

        print("=== Cleaning complete ===")

    elif args.step == "sentiment":
        if not (args.cleaned_pros and args.cleaned_cons):
            raise SystemExit("For step=sentiment, you must provide --cleaned-pros and --cleaned-cons")

        pros_tokens = read_token_lines(Path(args.cleaned_pros))
        cons_tokens = read_token_lines(Path(args.cleaned_cons))

        pros_chunks = chunk_tokens(pros_tokens, chunk_size=args.chunk_size, overlap=args.overlap)
        cons_chunks = chunk_tokens(cons_tokens, chunk_size=args.chunk_size, overlap=args.overlap)

        pros_rows = analyze_sentiment_chunks(pros_chunks)
        cons_rows = analyze_sentiment_chunks(cons_chunks)

        with (outdir / "sentiment_chunks_pros.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_id","n_tokens","neg","neu","pos","compound"])
            writer.writeheader()
            writer.writerows(pros_rows)
        with (outdir / "sentiment_chunks_cons.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_id","n_tokens","neg","neu","pos","compound"])
            writer.writeheader()
            writer.writerows(cons_rows)

        def summarize(rows: List[Dict]) -> Dict:
            if not rows:
                return {"n_chunks": 0, "mean": {}, "weighted": {}}
            mean = {
                "neg": sum(r["neg"] for r in rows) / len(rows),
                "neu": sum(r["neu"] for r in rows) / len(rows),
                "pos": sum(r["pos"] for r in rows) / len(rows),
                "compound": sum(r["compound"] for r in rows) / len(rows),
            }
            total = sum(r["n_tokens"] for r in rows)
            weighted = {
                "neg": sum(r["neg"]*r["n_tokens"] for r in rows) / total,
                "neu": sum(r["neu"]*r["n_tokens"] for r in rows) / total,
                "pos": sum(r["pos"]*r["n_tokens"] for r in rows) / total,
                "compound": sum(r["compound"]*r["n_tokens"] for r in rows) / total,
            }
            return {"n_chunks": len(rows), "mean": mean, "weighted": weighted}

        summary = {
            "params": {"chunk_size": args.chunk_size, "overlap": args.overlap},
            "pros": summarize(pros_rows),
            "cons": summarize(cons_rows)
        }
        (outdir / "sentiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("=== Sentiment computed and files written ===")

    elif args.step == "bigrams":
        if not (args.cleaned_pros and args.cleaned_cons):
            raise SystemExit("For step=bigrams, you must provide --cleaned-pros and --cleaned-cons")

        pros_tokens = read_token_lines(Path(args.cleaned_pros))
        cons_tokens = read_token_lines(Path(args.cleaned_cons))

        pros_rows = bigram_stats(pros_tokens, min_count=args.min_count)
        cons_rows = bigram_stats(cons_tokens, min_count=args.min_count)

        write_bigram_csv(pros_rows, outdir / "bigrams_pros.csv")
        write_bigram_csv(cons_rows, outdir / "bigrams_cons.csv")

        # Console summary (top 20 by count)
        def top_by_count(rows: List[Dict], k: int = 20) -> List[Dict]:
            return sorted(rows, key=lambda r: (r["count"], r["pmi"]), reverse=True)[:k]

        print("=== Bigrams complete ===")
        print("Top PROS:")
        for r in top_by_count(pros_rows, 20):
            print(f"{r['w1']} {r['w2']}  | count={r['count']}  pmi={r['pmi']:.2f}  t={r['t_score']:.2f}  dice={r['dice']:.3f}")
        print("\nTop CONS:")
        for r in top_by_count(cons_rows, 20):
            print(f"{r['w1']} {r['w2']}  | count={r['count']}  pmi={r['pmi']:.2f}  t={r['t_score']:.2f}  dice={r['dice']:.3f}")


if __name__ == "__main__":
    main()
