#!/usr/bin/env python
import argparse
from pathlib import Path
from datetime import datetime

from markitdown import MarkItDown  # pip install markitdown

def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF/DOCX/HTML/etc to Markdown using markitdown."
    )
    parser.add_argument("--input-file", required=True, help="Path to source document")
    parser.add_argument("--out-md", required=False, help="Output Markdown file")
    args = parser.parse_args()

    in_path = Path(args.input_file)
    if args.out_md:
        out_path = Path(args.out_md)
    else:
        out_path = in_path.with_suffix(".md")

    md = MarkItDown()
    result = md.convert(str(in_path))

    header = [
        "---",
        f'title: "{in_path.stem}"',
        f'source: "{in_path.name}"',
        f"converted: {datetime.now().isoformat(timespec='seconds')}",
        "---",
        "",
    ]
    md_text = "\n".join(header) + result.text_content.strip() + "\n"
    out_path.write_text(md_text, encoding="utf-8")
    print(f"Converted {in_path} → {out_path}")

if __name__ == "__main__":
    main()
