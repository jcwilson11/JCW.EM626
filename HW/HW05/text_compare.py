
#!/usr/bin/env python3
"""
Text comparison pipeline (Pro vs Con) — Steps 1-4

1. clean
2. sentiment
3. bigrams
4. lexdiv
"""

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Set, Dict, Tuple

# ---------- Common helpers ----------

URL_PATTERN = re.compile(
    r"(https?://\S+)|"
    r"(www\.\S+)|"
    r"([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})"
)
NUMBER_PATTERN = re.compile(r"\b\d+(?:[\.,]\d+)*\b")
PUNCT_AND_SYMBOLS = re.compile(r"[^\w\s']+", flags=re.UNICODE)
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


def read_token_lines(fp: Path) -> List[str]:
    return [line.strip() for line in fp.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]


# ---------- Step 2: sentiment helpers ----------

def chunk_tokens(tokens: List[str], chunk_size: int = 100, overlap: int = 0) -> List[List[str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")
    chunks: List[List[str]] = []
    step = chunk_size - overlap
    i = 0
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


# ---------- Step 3: bigrams ----------

def bigram_counts(tokens: List[str]) -> Tuple[Counter, Counter]:
    uni = Counter(tokens)
    bi = Counter()
    for i in range(len(tokens) - 1):
        bi[(tokens[i], tokens[i+1])] += 1
    return uni, bi


def bigram_stats(tokens: List[str], min_count: int = 2) -> List[Dict]:
    uni, bi = bigram_counts(tokens)
    N = len(tokens)
    B = max(1, N - 1)
    rows: List[Dict] = []
    for (w1, w2), c12 in bi.items():
        if c12 < min_count:
            continue
        c1 = uni[w1]
        c2 = uni[w2]
        pmi = math.log2((c12 * N) / (c1 * c2)) if c1 > 0 and c2 > 0 and c12 > 0 else float("-inf")
        expected = (c1 * c2) / B
        tscore = (c12 - expected) / math.sqrt(c12) if c12 > 0 else 0.0
        dice = (2 * c12) / (c1 + c2)
        rows.append({
            "w1": w1, "w2": w2,
            "count": c12,
            "left_count": c1, "right_count": c2,
            "pmi": pmi,
            "t_score": tscore,
            "dice": dice
        })
    rows.sort(key=lambda r: (r["count"], r["pmi"]), reverse=True)
    return rows


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Compare Pro/Con texts. Steps 1-4")
    parser.add_argument("--step", type=str, required=True, choices=["clean","sentiment","bigrams","lexdiv"])

    # cleaning
    parser.add_argument("--pros", type=str)
    parser.add_argument("--cons", type=str)
    parser.add_argument("--stopwords", type=str)
    parser.add_argument("--domain-stopwords", nargs="*", default=["gig","economy"])
    parser.add_argument("--outdir", type=str, default="")

    # steps 2-4
    parser.add_argument("--cleaned-pros", type=str)
    parser.add_argument("--cleaned-cons", type=str)

    # sentiment params
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--overlap", type=int, default=0)

    # bigrams params
    parser.add_argument("--min-count", type=int, default=2)

    args = parser.parse_args()

    # outdir resolution
    if args.outdir:
        outdir = Path(args.outdir).expanduser().resolve()
    else:
        # try infer from given files
        base = args.pros or args.cleaned_pros or "."
        base = Path(base).expanduser()
        outdir = base.parent.resolve() if base.is_file() else base.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if args.step == "clean":
        if not (args.pros and args.cons and args.stopwords):
            raise SystemExit("For step=clean, provide --pros, --cons, --stopwords")
        stopwords = load_stopwords(Path(args.stopwords))
        domain = set(args.domain_stopwords)
        pros_tokens = clean_file(Path(args.pros), stopwords, domain)
        cons_tokens = clean_file(Path(args.cons), stopwords, domain)
        (outdir / "cleaned_pros.txt").write_text("\n".join(pros_tokens) + "\n", encoding="utf-8")
        (outdir / "cleaned_cons.txt").write_text("\n".join(cons_tokens) + "\n", encoding="utf-8")
        # freq tables
        def write_freq(tokens, op):
            cnt = Counter(tokens)
            with op.open("w", encoding="utf-8") as f:
                f.write("token,count\n")
                for tok, c in cnt.most_common():
                    f.write(f"{tok},{c}\n")
        write_freq(pros_tokens, outdir / "word_freq_pros.csv")
        write_freq(cons_tokens, outdir / "word_freq_cons.csv")
        print("=== Cleaning complete ===")

    elif args.step == "sentiment":
        if not (args.cleaned_pros and args.cleaned_cons):
            raise SystemExit("For step=sentiment, provide --cleaned-pros and --cleaned-cons")
        pros_tokens = read_token_lines(Path(args.cleaned_pros))
        cons_tokens = read_token_lines(Path(args.cleaned_cons))
        pros_chunks = chunk_tokens(pros_tokens, args.chunk_size, args.overlap)
        cons_chunks = chunk_tokens(cons_tokens, args.chunk_size, args.overlap)
        pros_rows = analyze_sentiment_chunks(pros_chunks)
        cons_rows = analyze_sentiment_chunks(cons_chunks)
        # write CSVs
        with (outdir / "sentiment_chunks_pros.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_id","n_tokens","neg","neu","pos","compound"])
            writer.writeheader()
            writer.writerows(pros_rows)
        with (outdir / "sentiment_chunks_cons.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["chunk_id","n_tokens","neg","neu","pos","compound"])
            writer.writeheader()
            writer.writerows(cons_rows)
        summary = {
            "params": {"chunk_size": args.chunk_size, "overlap": args.overlap},
            "pros": summarize_sentiment(pros_rows),
            "cons": summarize_sentiment(cons_rows)
        }
        (outdir / "sentiment_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("=== Sentiment done ===")

    elif args.step == "bigrams":
        if not (args.cleaned_pros and args.cleaned_cons):
            raise SystemExit("For step=bigrams, provide --cleaned-pros and --cleaned-cons")
        pros_tokens = read_token_lines(Path(args.cleaned_pros))
        cons_tokens = read_token_lines(Path(args.cleaned_cons))
        pros_rows = bigram_stats(pros_tokens, min_count=args.min_count)
        cons_rows = bigram_stats(cons_tokens, min_count=args.min_count)
        # write
        with (outdir / "bigrams_pros.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["w1","w2","count","left_count","right_count","pmi","t_score","dice"])
            writer.writeheader()
            writer.writerows(pros_rows)
        with (outdir / "bigrams_cons.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["w1","w2","count","left_count","right_count","pmi","t_score","dice"])
            writer.writeheader()
            writer.writerows(cons_rows)
        print("=== Bigrams done ===")

    elif args.step == "lexdiv":
        if not (args.cleaned_pros and args.cleaned_cons):
            raise SystemExit("For step=lexdiv, provide --cleaned-pros and --cleaned-cons")
        pros_tokens = read_token_lines(Path(args.cleaned_pros))
        cons_tokens = read_token_lines(Path(args.cleaned_cons))

        def lexdiv(tokens: List[str]) -> Dict[str, float]:
            total = len(tokens)
            uniq = len(set(tokens))
            ratio = (uniq / total) if total else 0.0
            return {"total_tokens": total, "unique_tokens": uniq, "lexical_diversity": ratio}

        pros_ld = lexdiv(pros_tokens)
        cons_ld = lexdiv(cons_tokens)

        out = {
            "pros": pros_ld,
            "cons": cons_ld,
            "diff_cons_minus_pros": cons_ld["lexical_diversity"] - pros_ld["lexical_diversity"]
        }
        (outdir / "lexical_diversity.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

        # CSV
        with (outdir / "lexical_diversity.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["side","total_tokens","unique_tokens","lexical_diversity"])
            writer.writeheader()
            writer.writerow({"side": "pros", **pros_ld})
            writer.writerow({"side": "cons", **cons_ld})

        print("=== Lexical diversity (cleaned) ===")
        print(f"PROS: total={pros_ld['total_tokens']} unique={pros_ld['unique_tokens']} ratio={pros_ld['lexical_diversity']:.4f}")
        print(f"CONS: total={cons_ld['total_tokens']} unique={cons_ld['unique_tokens']} ratio={cons_ld['lexical_diversity']:.4f}")
        print(f"(CONS - PROS) ratio = {out['diff_cons_minus_pros']:.4f}")


if __name__ == "__main__":
    main()
