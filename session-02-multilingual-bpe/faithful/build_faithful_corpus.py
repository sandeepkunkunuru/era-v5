#!/usr/bin/env python3
"""Fetch the India Wikipedia pages (en/hi/te/mai) and convert them to a *faithful*
Markdown corpus — the input the ERA V5 Assignment-2 grader actually evaluates against.

"Faithful" = keep every visible article artifact the HTML→Markdown converter emits:
links, URLs, tables, references, image links, navboxes, categories. Only script/style/meta
machinery is stripped. This is what makes the round-trip gate meaningful: the corpus is full
of `[`, `]`, `|`, `#`, `_`, `/`, `:` from Markdown, so a tokenizer that drops any of them fails.

Method matches the published reference solution (reference/extracted/build_wiki_faithful_markdown.py)
exactly, so our regenerated corpus is directly comparable. Wikipedia pages drift over time, so the
committed corpus/ snapshot is the authoritative one for our reported numbers.

    python build_faithful_corpus.py
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from urllib.parse import quote, urljoin

import regex
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

OUT = Path(__file__).resolve().parent / "corpus"
USER_AGENT = "era-v5-faithful-tokenizer/1.0 (coursework; contact via TSAI)"
FAITHFUL_UNIT_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+|[^\s\p{L}\p{M}\p{N}]")

PAGES = {
    "en": ("English", "India"),
    "hi": ("Hindi", "भारत"),
    "te": ("Telugu", "భారతదేశం"),
    "mai": ("Maithili", "भारत"),
}


def faithful_units(text: str) -> int:
    return len(FAITHFUL_UNIT_RE.findall(text))


def get(url: str) -> requests.Response:
    return requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=(8, 30))


def absolutize_links(soup: BeautifulSoup, lang: str) -> None:
    base = f"https://{lang}.wikipedia.org/wiki/"
    for tag in soup.find_all(["a", "img", "source"]):
        attr = "href" if tag.name == "a" else "src"
        value = tag.get(attr)
        if not value:
            continue
        if value.startswith("//"):
            tag[attr] = "https:" + value
        elif value.startswith("./"):
            tag[attr] = urljoin(base, value[2:])
        elif value.startswith("/"):
            tag[attr] = urljoin(f"https://{lang}.wikipedia.org", value)


def strip_only_technical_noise(node: BeautifulSoup, soup: BeautifulSoup) -> None:
    for tag in node(["script", "style", "meta"]):
        tag.decompose()
    for tag in node.find_all("link"):
        rel = " ".join(tag.get("rel") or [])
        href = tag.get("href") or ""
        if "mw:PageProp/Category" in rel and href:
            tag.replace_with(soup.new_string(f"\nCategory: {href}\n"))
        else:
            tag.decompose()


def normalize_markdown(markdown: str) -> str:
    markdown = markdown.replace("\xa0", " ")
    markdown = re.sub(r"\n{4,}", "\n\n\n", markdown)
    markdown = re.sub(r"[ \t]+\n", "\n", markdown)
    return markdown.strip() + "\n"


def build_one(lang: str, title: str) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    url = f"https://{lang}.wikipedia.org/api/rest_v1/page/html/{quote(title)}"

    res = get(url)
    res.raise_for_status()
    (OUT / f"{lang}.raw.html").write_text(res.text, encoding="utf-8")

    soup = BeautifulSoup(res.text, "lxml")
    body = soup.find("body") or soup
    strip_only_technical_noise(body, soup)
    absolutize_links(body, lang)
    markdown = normalize_markdown(
        md(str(body), heading_style="ATX", bullets="-", strip=["span"])
    )

    (OUT / f"{lang}.faithful.md").write_text(markdown, encoding="utf-8")
    (OUT / f"{lang}.faithful.txt").write_text(markdown, encoding="utf-8")
    meta = {
        "lang": lang,
        "title": title,
        "source_url": url,
        "variant": "wiki_faithful_markdown",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "chars": len(markdown),
        "faithful_units": faithful_units(markdown),
        "unit_policy": "contiguous L/M/N run = 1 unit; each visible non-space punctuation/symbol = 1 unit",
    }
    (OUT / f"{lang}.meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> int:
    for code, (name, title) in PAGES.items():
        meta = build_one(code, title)
        print(f"{code:4s} {name:9s} {meta['faithful_units']:>7} faithful units  ({meta['chars']} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
