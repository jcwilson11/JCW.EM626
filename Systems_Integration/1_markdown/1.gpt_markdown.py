import argparse
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="Wrap a ChatGPT-generated system manual into a Markdown file."
    )
    parser.add_argument("--input-txt", required=True, help="Plain text file copied from ChatGPT")
    parser.add_argument("--out-md", required=True, help="Output Markdown file")
    parser.add_argument("--title", required=False, default="Untitled System Manual")
    parser.add_argument("--source", required=False, default="ChatGPT")
    args = parser.parse_args()

    raw = Path(args.input_txt).read_text(encoding="utf-8")

    header = [
        "---",
        f'title: "{args.title}"',
        f'source: "{args.source}"',
        f"created: {datetime.now().isoformat(timespec='seconds')}",
        "---",
        "",
    ]
    md_text = "\n".join(header) + raw.strip() + "\n"
    Path(args.out_md).write_text(md_text, encoding="utf-8")
    print(f"Wrote Markdown manual to {args.out_md}")

if __name__ == "__main__":
    main()
