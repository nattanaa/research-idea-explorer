"""Search papers by meaning with the Semantic Scholar Graph and Recommendations APIs.

Adds what OpenAlex lacks: abstracts and TLDR summaries for most papers, and a
"similar papers" endpoint. Author ids here are Semantic Scholar's, so papers.py looks
every paper up in OpenAlex again before the authors are used for the expert ranking.

S2_API_KEY in .env is optional but strongly recommended: with a key you get 1 request
per second of your own; without one you share a pool that is often rate-limited.
"""

import os
import re
from pathlib import Path

import httpx
from dotenv import load_dotenv

from _http import Client

load_dotenv(Path(__file__).parent / ".env")

BASE = "https://api.semanticscholar.org"
FIELDS = "title,year,abstract,tldr,citationCount,externalIds,authors,url"
MAX_AUTHORS = 10
# The recommendations endpoint rejects "tldr" (400: unsupported fields), so it gets its own list
RECOMMEND_FIELDS = FIELDS.replace("tldr,", "")
S2_ID = re.compile(r"[0-9a-f]{40}")  # a Semantic Scholar paper id

# 1 call per second with a key (1.5 s: several Researchers share this limit, so keep a wide margin). Without one: about 100 calls per 5 minutes.
_client = Client("semantic_scholar", min_interval=1.5 if os.environ.get("S2_API_KEY") else 3.2, timeout=20, attempts=3)


def _headers() -> dict:
    key = os.environ.get("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def _normalize(p: dict) -> dict:
    ext = p.get("externalIds") or {}
    return {
        "source": "semantic_scholar",
        "source_id": p.get("paperId"),
        "title": p.get("title"),
        "year": p.get("year"),
        "abstract": p.get("abstract") or "",
        "tldr": (p.get("tldr") or {}).get("text") or "",
        "citations": p.get("citationCount"),
        "doi": ext.get("DOI"),
        "pmid": ext.get("PubMed"),
        "arxiv_id": ext.get("ArXiv"),
        "authors": [{"id": None, "name": a.get("name")} for a in (p.get("authors") or [])[:MAX_AUTHORS]],
        "url": p.get("url"),
    }


def search(query: str, from_year: int | None = None, limit: int = 10) -> list[dict]:
    """Papers matching a query, ranked by Semantic Scholar's relevance model."""
    params = {"query": query, "limit": min(limit, 25), "fields": FIELDS}
    if from_year:
        params["year"] = f"{from_year}-"
    data = _client.get(BASE + "/graph/v1/paper/search", params, headers=_headers())
    return [_normalize(p) for p in data.get("data") or [] if p.get("title")]


def get_paper(identifier: str) -> dict | None:
    """One paper. identifier: 'DOI:10.1038/...', 'ARXIV:1706.03762', 'PMID:123' or a paper id."""
    try:
        return _normalize(_client.get(f"{BASE}/graph/v1/paper/{identifier}", {"fields": FIELDS}, headers=_headers()))
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise


def get_similar(identifier: str, limit: int = 10) -> list[dict]:
    """Papers similar to this one, by Semantic Scholar's recommendations.
    The recommendations endpoint wants Semantic Scholar's own paper id, not 'DOI:...' (that gave a 400),
    so any other kind of id is looked up first."""
    if not S2_ID.fullmatch(identifier):
        paper = get_paper(identifier)
        if not paper:
            return []
        identifier = paper["source_id"]
    try:
        data = _client.get(
            f"{BASE}/recommendations/v1/papers/forpaper/{identifier}",
            {"limit": min(limit, 25), "fields": RECOMMEND_FIELDS},
            headers=_headers(),
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return []
        raise
    return [_normalize(p) for p in data.get("recommendedPapers") or [] if p.get("title")]


if __name__ == "__main__":
    # Smoke test: uv run semantic_scholar.py "graph neural networks for drug discovery"
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "graph neural networks for drug discovery"
    print("API key:", "yes" if os.environ.get("S2_API_KEY") else "no (shared pool, may be rate-limited)")
    results = search(query, limit=5)
    assert results, "search returned nothing"
    for r in results:
        print(f"{r['year']}  {r['citations']:>6} cites  {r['title']}")
        print("   TLDR:", r["tldr"][:100] or "(none)")
    similar = get_similar(results[0]["source_id"], limit=3)
    print("\nSimilar to the first:", [s["title"] for s in similar])
    print("\nok")
