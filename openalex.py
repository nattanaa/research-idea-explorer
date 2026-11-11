"""Search and read research papers through the OpenAlex API.

Replaces wikipedia.py. Gives the agents three things to walk the citation graph:
  search_works(query)      find papers by keyword
  get_work(id)             one paper: abstract, authors, the ids of the papers it cites
  get_references(id)       the papers this one cites (resolved to titles)
  get_citing(id)           the papers that cite this one

Credits (OpenAlex bills per call; check openalex.org for current numbers):
  fetching one paper by id is free, search and list calls cost 1 credit each.
So responses are cached on disk: asking the same thing twice costs nothing.

Needs OPENALEX_API_KEY in .env (free key: openalex.org/settings/api).
"""

import hashlib
import json
import os
import threading
import time
from pathlib import Path

import httpx
from dotenv import load_dotenv

from _http import check

load_dotenv(Path(__file__).parent / ".env")

BASE = "https://api.openalex.org"
CACHE_DIR = Path(__file__).parent / ".cache" / "openalex"
MIN_INTERVAL = 0.15  # seconds between calls, shared by every thread (researchers run in parallel)
MAX_AUTHORS = 10  # authors kept per paper: enough to find the lead and senior ones

# Only ask OpenAlex for the fields we use: smaller responses, faster calls.
LIST_FIELDS = "id,display_name,publication_year,cited_by_count,authorships,type,doi,primary_topic"
WORK_FIELDS = LIST_FIELDS + ",doi,abstract_inverted_index,referenced_works"

_lock = threading.Lock()
_last_call = 0.0


def _request(path: str, params: dict) -> dict:
    """GET one OpenAlex endpoint, with a disk cache, a shared rate limit and retries."""
    global _last_call
    cache_key = hashlib.sha1((path + json.dumps(params, sort_keys=True)).encode()).hexdigest()
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    api_key = os.environ.get("OPENALEX_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENALEX_API_KEY in .env (free key: openalex.org/settings/api).")

    for attempt in range(3):
        with _lock:  # one call at a time, MIN_INTERVAL apart
            wait = _last_call + MIN_INTERVAL - time.time()
            if wait > 0:
                time.sleep(wait)
            _last_call = time.time()
        response = httpx.get(BASE + path, params={**params, "api_key": api_key}, timeout=30)
        if response.status_code in (429, 500, 502, 503, 504) and attempt < 2:
            time.sleep(2 ** (attempt + 1))  # back off: 2 s, then 4 s
            continue
        check(response)  # not raise_for_status: its message would print the URL, which carries the key
        data = response.json()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(data))
        return data
    raise RuntimeError("unreachable")


def short_id(value: str) -> str:
    """'https://openalex.org/W2741809807' -> 'W2741809807'."""
    return value.rsplit("/", 1)[-1]


def rebuild_abstract(inverted: dict | None) -> str:
    """OpenAlex stores abstracts as {word: [positions]}. Put the words back in order."""
    if not inverted:
        return ""
    words = {pos: word for word, positions in inverted.items() for pos in positions}
    return " ".join(words[i] for i in sorted(words))


def _authors(work: dict) -> list[dict]:
    """Authors with their ids, so the same person is never counted twice under two spellings."""
    authors = []
    authorships = work.get("authorships", [])
    if len(authorships) > MAX_AUTHORS:
        # Keep the last few too: the last author is often the senior researcher
        authorships = authorships[: MAX_AUTHORS - 3] + authorships[-3:]
    for a in authorships:
        person = a.get("author") or {}
        if not person.get("id"):
            continue
        authors.append({
            "id": short_id(person["id"]),
            "name": person.get("display_name"),
            "orcid": person.get("orcid"),
            "position": a.get("author_position"),  # first | middle | last
            "corresponding": bool(a.get("is_corresponding")),
            "institutions": [i["display_name"] for i in a.get("institutions", []) if i.get("display_name")],
        })
    return authors


def _brief(work: dict) -> dict:
    return {
        "id": short_id(work["id"]),
        "title": work.get("display_name"),
        "year": work.get("publication_year"),
        "citations": work.get("cited_by_count", 0),
        "authors": _authors(work),
        "url": work["id"],
        "doi": work.get("doi"),
        # The broad field (Medicine, Computer Science...): papers.py uses it to pick sources
        "field": ((work.get("primary_topic") or {}).get("field") or {}).get("display_name"),
    }


def search_works(query: str, from_year: int | None = None, limit: int = 10) -> list[dict]:
    """Papers matching a keyword search, most relevant first. 1 credit."""
    params = {"search": query, "per-page": min(limit, 25), "select": LIST_FIELDS}
    if from_year:
        params["filter"] = f"from_publication_date:{from_year}-01-01"
    return [_brief(w) for w in _request("/works", params).get("results", [])]


def get_work(work_id: str) -> dict:
    """One paper with its abstract and the ids of the papers it cites. Free."""
    work = _request(f"/works/{short_id(work_id)}", {"select": WORK_FIELDS})
    return {
        **_brief(work),
        "doi": work.get("doi"),
        "abstract": rebuild_abstract(work.get("abstract_inverted_index")),
        "reference_ids": [short_id(r) for r in work.get("referenced_works", [])],
    }


def resolve(doi: str | None = None, pmid: str | None = None, arxiv_id: str | None = None) -> dict | None:
    """Find a paper that another source returned, by its DOI, PubMed id or arXiv id.

    Gives the paper's OpenAlex record, so its authors carry OpenAlex ids. Free (a single
    lookup). None if OpenAlex does not know the paper."""
    candidates = []
    if doi:
        candidates.append("doi:" + doi.removeprefix("https://doi.org/"))
    if pmid:
        candidates.append(f"pmid:{pmid}")
    if arxiv_id:
        candidates.append(f"doi:10.48550/arXiv.{arxiv_id}")  # arXiv's own DOIs
    for external_id in candidates:
        try:
            return _brief(_request(f"/works/{external_id}", {"select": LIST_FIELDS}))
        except httpx.HTTPStatusError as e:
            if e.response.status_code != 404:
                raise
    return None


def get_references(work_id: str, limit: int = 15) -> list[dict]:
    """The papers this one cites, most cited first. 1 credit (plus the free lookup of the paper)."""
    ids = get_work(work_id)["reference_ids"][:50]  # a filter takes up to 50 values
    if not ids:
        return []
    params = {
        "filter": "openalex:" + "|".join(ids),
        "per-page": min(limit, 25),
        "sort": "cited_by_count:desc",
        "select": LIST_FIELDS,
    }
    return [_brief(w) for w in _request("/works", params).get("results", [])]


def get_citing(work_id: str, limit: int = 15) -> list[dict]:
    """Papers that cite this one, most cited first. 1 credit."""
    params = {
        "filter": f"cites:{short_id(work_id)}",
        "per-page": min(limit, 25),
        "sort": "cited_by_count:desc",
        "select": LIST_FIELDS,
    }
    return [_brief(w) for w in _request("/works", params).get("results", [])]


if __name__ == "__main__":
    # Smoke test: uv run openalex.py "graph neural networks for drug discovery"
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "graph neural networks for drug discovery"
    results = search_works(query, limit=5)
    assert results, "search returned nothing: check OPENALEX_API_KEY"
    for w in results:
        print(f"{w['id']}  {w['year']}  {w['citations']:>6} cites  {w['title']}")

    first = get_work(results[0]["id"])
    print("\nFirst paper:", first["title"])
    print("Abstract:", first["abstract"][:200] or "(none)")
    print("Authors:", [(a["name"], a["institutions"][:1]) for a in first["authors"][:3]])
    print("Cites", len(first["reference_ids"]), "papers")

    refs = get_references(first["id"], limit=3)
    print("\nTop references:", [r["title"] for r in refs])
    citing = get_citing(first["id"], limit=3)
    print("Top citing papers:", [c["title"] for c in citing])
    print("\nok")
