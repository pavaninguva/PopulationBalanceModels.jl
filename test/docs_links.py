"""Check rendered links/assets and canonical API-index coverage after docs/make.jl.

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
        self.bindings = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        self.links.extend(attrs[key] for key in ("href", "src") if key in attrs)
        if tag == "a" and "docstring-binding" in attrs.get("class", "").split():
            self.bindings.append(unquote(urlsplit(attrs["href"]).fragment))


def check_api(root, pages):
    """Check the rendered API catalogue, independently of its source-file router."""
    index = "api/index.html" if "api/index.html" in pages else "api.html"
    if index not in pages:
        return [(index, "", "missing API index")]
    errors = []
    owners = {}
    for name, page in pages.items():
        for binding in page.bindings:
            if not name.startswith("api/") or name == index:
                errors.append((name, binding, "API docstring belongs on a topic reference page"))
            if binding in owners:
                errors.append((name, binding, "duplicate canonical API entry"))
            owners[binding] = name
    if not owners:
        errors.append((index, "", "empty API catalogue"))
    links = set()
    for href in pages[index].links:
        url = urlsplit(href)
        if url.scheme or url.netloc or not url.fragment:
            continue
        target = posixpath.normpath(posixpath.join(posixpath.dirname(index), unquote(url.path)))
        if target not in pages:
            target = posixpath.join(target, "index.html")
        links.add((target, unquote(url.fragment)))
    for binding, page in owners.items():
        if (page, binding) not in links:
            errors.append((index, binding, "canonical API entry missing from symbol index"))
    guide = ("uncertainty_quantification/index.html" if
             "uncertainty_quantification/index.html" in pages else "uncertainty_quantification.html")
    if guide not in pages or "Forward-UQ-reference" not in pages[guide].ids:
        errors.append((guide, "Forward-UQ-reference", "missing legacy UQ guide anchor"))
    reference_pages = {index, *owners.values()}
    largest = max(reference_pages, key=lambda page: (root / page).stat().st_size)
    size = (root / largest).stat().st_size
    if size > 200 * 1024:
        errors.append((largest, "", "API page exceeds 200 KiB; split the topic"))
    print(f"API: {len(owners)} canonical entries on {len(reference_pages)-1} topic pages; "
          f"largest {largest}: {size/1024:.1f} KiB")
    return errors


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
    errors, checked, deployment = check_api(root, pages), 0, 0
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
