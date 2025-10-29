#!/usr/bin/env python3
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

URL = "https://www.britannica.com/procon/gig-economy-debate"

HEAD_RE = re.compile(r"^\s*(Pro|Con)\s+\d+\s*:\s*", re.I)
HEADING_TAGS = {"h2", "h3", "h4"}
STOP_SECTION_RE = re.compile(
    r"^\s*(Pros and Cons at a Glance|Discussion Questions|Take Action|Sources)\b",
    re.I,
)

def fetch(url: str) -> str:
    r = requests.get(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
            )
        },
        timeout=30,
    )
    r.raise_for_status()
    return r.text

def clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def collect_after_heading(h: Tag) -> str:
    """Collect paragraphs/lists/quotes after a Pro/Con heading until the next heading or end section."""
    chunks = []
    for sib in h.next_siblings:
        if isinstance(sib, Tag):
            text = sib.get_text(" ", strip=True)

            # Stop if we hit another heading (Pro/Con or a new big section)
            if sib.name in HEADING_TAGS:
                if HEAD_RE.match(text) or STOP_SECTION_RE.match(text):
                    break

            # Gather main content pieces
            if sib.name in {"p"}:
                t = clean(sib.get_text(" ", strip=True))
                if t:
                    chunks.append(t)
            elif sib.name in {"ul", "ol"}:
                items = []
                for li in sib.find_all("li", recursive=False):
                    t = clean(li.get_text(" ", strip=True))
                    if t:
                        items.append(f"- {t}")
                if items:
                    chunks.append("\n".join(items))
            elif sib.name == "blockquote":
                t = clean(sib.get_text(" ", strip=True))
                if t:
                    chunks.append(t)

    body = "\n\n".join(chunks).strip()
    return body

def extract_procons(html: str):
    soup = BeautifulSoup(html, "html.parser")

    # Find all real article headings like "Pro 1: ...", "Con 2: ..."
    heads = []
    for tag in soup.find_all(HEADING_TAGS):
        txt = tag.get_text(" ", strip=True)
        if HEAD_RE.match(txt):
            heads.append(tag)

    pros, cons = [], []

    for h in heads:
        heading_text = clean(h.get_text(" ", strip=True))
        body = collect_after_heading(h)

        # Skip the "at a glance" rows that only have "Read More."
        if not body or body.lower() == "read more.":
            continue

        full_entry = f"{heading_text}\n{body}".strip()
        if heading_text.lower().startswith("pro"):
            pros.append(full_entry)
        else:
            cons.append(full_entry)

    # Dedupe & keep order (in case of accidental repeats)
    def dedupe(seq):
        seen = set()
        out = []
        for x in seq:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    return dedupe(pros), dedupe(cons)

def save(filename: str, lines):
    Path(filename).write_text("\n\n" + ("\n\n".join(lines)) + "\n", encoding="utf-8")

def main():
    html = fetch(URL)
    pros, cons = extract_procons(html)

    if not pros and not cons:
        raise SystemExit("No expanded Pro/Con entries found. The page structure may have changed.")

    save("pros.txt", pros)
    save("cons.txt", cons)

    print(f"Saved {len(pros)} pros -> pros.txt and {len(cons)} cons -> cons.txt")

if __name__ == "__main__":
    main()
