"""Search preprints on arXiv (computer science, physics, math, statistics...).

arXiv's rules: one request every 3 seconds, on a single connection, shared by everything
you run. The client in _http.py enforces both, so a search takes at least 3 seconds and
parallel Researchers queue up. No key is needed.

arXiv has no citation counts and no author ids: papers.py looks each paper up in
OpenAlex afterwards.
"""

import re
import xml.etree.ElementTree as ET

from _http import Client

URL = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
MAX_AUTHORS = 10
MAX_TERMS = 8  # a long query with every word required matches nothing
# Filler words are dropped: arXiv matches words literally, and "for" or "of" would rule papers out
STOPWORDS = {
    "a", "an", "the", "for", "of", "in", "on", "at", "to", "by", "with", "from", "and", "or",
    "using", "use", "via", "is", "are", "be", "as", "into", "that", "this", "their", "its", "about",
}

_client = Client("arxiv", min_interval=3.1, serialize=True, timeout=20, attempts=2)  # optional source: fail fast


def _clean(text: str) -> str:
    return " ".join(text.split())  # arXiv titles and abstracts contain hard line breaks


def parse_feed(xml_text: str) -> list[dict]:
    """arXiv Atom feed -> one normalized record per paper."""
    papers = []
    for entry in ET.fromstring(xml_text).findall("a:entry", NS):
        link = entry.findtext("a:id", "", NS)
        if "/api/errors" in link:  # a bad query comes back as a fake entry
            continue
        arxiv_id = re.sub(r"v\d+$", "", link.rsplit("/abs/", 1)[-1])
        published = entry.findtext("a:published", "", NS)
        papers.append({
            "source": "arxiv",
            "source_id": arxiv_id,
            "title": _clean(entry.findtext("a:title", "", NS)),
            "year": int(published[:4]) if published else None,
            "abstract": _clean(entry.findtext("a:summary", "", NS)),
            "tldr": "",
            "citations": None,
            "doi": entry.findtext("x:doi", None, NS),
            "pmid": None,
            "arxiv_id": arxiv_id,
            "authors": [{"id": None, "name": _clean(a.findtext("a:name", "", NS))} for a in entry.findall("a:author", NS)[:MAX_AUTHORS]],
            "url": f"https://arxiv.org/abs/{arxiv_id}",
        })
    return papers


def build_query(query: str) -> str:
    """'graph neural networks for drug discovery' -> 'all:graph AND all:neural AND ...' (no 'for')."""
    words = [w for w in re.findall(r"[A-Za-z0-9]+", query) if w.lower() not in STOPWORDS][:MAX_TERMS]
    return " AND ".join(f"all:{w}" for w in words)


def search(query: str, from_year: int | None = None, limit: int = 10) -> list[dict]:
    """Papers whose title, abstract or authors contain all the query's words, most relevant first."""
    search_query = build_query(query)
    if not search_query:
        return []
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": min(limit, 25),
        "sortBy": "relevance",
    }
    papers = parse_feed(_client.get(URL, params, text=True))
    if from_year:  # the API has no simple year filter: drop older papers here
        papers = [p for p in papers if p["year"] and p["year"] >= from_year]
    return papers


if __name__ == "__main__":
    # Smoke test: uv run arxiv_source.py "graph neural networks drug discovery"   (takes a few seconds)
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "graph neural networks drug discovery"
    results = search(query, limit=5)
    assert results, "search returned nothing"
    for r in results:
        print(f"{r['year']}  arXiv:{r['arxiv_id']}  {r['title'][:90]}")
        print("   authors:", [a["name"] for a in r["authors"][:3]])
    print("\nok")
