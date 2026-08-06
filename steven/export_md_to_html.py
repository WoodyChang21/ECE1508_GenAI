"""Convert a markdown file (default: v1.md) to a standalone, self-contained HTML file.

Writes to the system temp directory by default -- NOT inside this repo -- so a stale
HTML export never accidentally lingers here after the source markdown changes. Pass
--out to choose a different destination instead.

Images referenced by relative path in the markdown are base64-embedded directly into
the HTML, so the exported file is fully portable (works from any location/machine,
no dependency on outputs/sample_plots/ sitting next to it).

Usage:
    python steven/export_md_to_html.py
    python steven/export_md_to_html.py --source steven/backlog.md --out ~/Desktop/backlog.html
"""

from __future__ import annotations

import argparse
import base64
import mimetypes
import re
import tempfile
from pathlib import Path

import markdown

HERE = Path(__file__).resolve().parent

STYLE = """
  body {
    max-width: 900px; margin: 2rem auto; padding: 0 1.5rem;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    line-height: 1.6; color: #1a1a1a;
  }
  h1, h2, h3 { line-height: 1.3; }
  h1 { border-bottom: 2px solid #ddd; padding-bottom: 0.5rem; }
  h2 { border-bottom: 1px solid #eee; padding-bottom: 0.3rem; margin-top: 2.5rem; }
  code { background: #f4f4f4; padding: 0.15em 0.4em; border-radius: 4px; font-size: 0.9em; }
  pre { background: #f4f4f4; padding: 1em; border-radius: 6px; overflow-x: auto; }
  pre code { background: none; padding: 0; }
  table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
  th, td { border: 1px solid #ddd; padding: 0.5em 0.8em; text-align: left; }
  th { background: #f4f4f4; }
  img { max-width: 100%; height: auto; border: 1px solid #eee; border-radius: 4px; margin: 0.5rem 0; }
  blockquote { border-left: 4px solid #ddd; margin-left: 0; padding-left: 1em; color: #555; }
  a { color: #0969da; }
"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=str, default=str(HERE / "v1.md"))
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output HTML path. Defaults to the system temp dir (not this repo).",
    )
    return p.parse_args()


def embed_images(html: str, base_dir: Path) -> str:
    """Replace <img src="relative/path.png"> with an inlined base64 data: URI, so the
    exported HTML has no dependency on the repo's file layout."""

    def repl(match: re.Match) -> str:
        src = match.group(1)
        if src.startswith(("http://", "https://", "data:")):
            return match.group(0)
        img_path = (base_dir / src).resolve()
        if not img_path.is_file():
            return match.group(0)  # leave as-is; broken links are visible, not silently swallowed
        mime, _ = mimetypes.guess_type(str(img_path))
        mime = mime or "application/octet-stream"
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        return match.group(0).replace(f'src="{src}"', f'src="data:{mime};base64,{b64}"')

    return re.sub(r'<img[^>]*src="([^"]+)"[^>]*>', repl, html)


def convert(source: Path) -> str:
    text = source.read_text()
    body = markdown.markdown(text, extensions=["tables", "fenced_code", "toc"])
    body = embed_images(body, source.parent)
    title = source.stem
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{STYLE}</style>
</head>
<body>
{body}
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    source = Path(args.source).resolve()
    out = Path(args.out).expanduser().resolve() if args.out else Path(tempfile.gettempdir()) / f"{source.stem}.html"

    html = convert(source)
    out.write_text(html)
    print(f"wrote {out} ({len(html)} bytes, images embedded inline)")


if __name__ == "__main__":
    main()
