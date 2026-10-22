"""Check local rendered-document links, images and anchors after docs/make.jl.

Documenter's deploy-time version metadata is absent from a local build. Remote
URLs are outside this check; the Documenter build checks its own cross-references.
"""
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit
import posixpath
import sys


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        self.links.extend(attrs[key] for key in ("href", "src") if key in attrs)


def check(root):
    files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    pages = {}
    for name in files:
        if name.endswith(".html"):
            page = Page()
            page.feed((root / name).read_text(encoding="utf-8"))
            pages[name] = page
    if not pages:
        raise RuntimeError(f"No HTML pages under {root}; build the docs first")
    errors, checked, deployment = [], 0, 0
    for name, page in pages.items():
        for link in page.links:
            url = urlsplit(link)
            if url.scheme or url.netloc or not link:
                continue
            target = posixpath.normpath(posixpath.join(posixpath.dirname(name), unquote(url.path))) if url.path else name
            index = posixpath.normpath(posixpath.join(target, "index.html"))
            if target not in files and index in files:
                target = index
            if target not in files:
                if posixpath.basename(target) in ("siteinfo.js", "versions.js"):
                    deployment += 1
                    continue
                errors.append((name, link, "missing file"))
                continue
            if url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:
                errors.append((name, link, "missing anchor"))
            checked += 1
    print(f"HTML pages: {len(pages)}; checked local links/assets: {checked}; "
          f"broken: {len(errors)}; deployment metadata: {deployment}")
    for error in errors:
        print(error)
    return not errors


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "docs" / "build"
    sys.exit(0 if check(root) else 1)
