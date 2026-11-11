"""One search over several sources. The agents' tools call this file, never a source directly.

    search_papers("idea")
       OpenAlex (always) + Semantic Scholar + PubMed (biomedical) + arXiv (CS, physics, math)
       -> merge and dedupe (same DOI, else same title)
       -> look every paper up in OpenAlex again, so its authors carry OpenAlex ids

Why the last step: each source numbers authors differently, and the same person would
count as three. The expert ranking only trusts OpenAlex author ids. A paper OpenAlex does
not know is kept, but its authors have no id and the ranking skips them.

One source failing (a rate limit, a timeout) never fails the search: it is reported in
"errors" and the rest still come back.
"""

import re
from concurrent.futures import ThreadPoolExecutor, wait

import openalex
import semantic_scholar
import pubmed
import arxiv_source

# OpenAlex's broad fields -> which extra sources are worth their rate limit
BIOMEDICAL = {
    "Medicine", "Biochemistry, Genetics and Molecular Biology", "Neuroscience",
    "Immunology and Microbiology", "Pharmacology, Toxicology and Pharmaceutics",
    "Health Professions", "Nursing", "Dentistry",
}
PREPRINT = {"Computer Science", "Mathematics", "Physics and Astronomy", "Decision Sciences", "Engineering"}

SOURCE_DEADLINE = 35  # seconds to wait for the extra sources together; a slow one is skipped, not waited for

SOURCES = {
    "semantic_scholar": semantic_scholar.search,
    "pubmed": pubmed.search,
    "arxiv": arxiv_source.search,
}


def norm_doi(doi: str | None) -> str | None:
    return doi.lower().removeprefix("https://doi.org/").strip() if doi else None


def _key(paper: dict) -> str:
    if paper.get("doi"):
        return "doi:" + paper["doi"]
    return "title:" + re.sub(r"[^a-z0-9]", "", (paper.get("title") or "").lower())


def _from_openalex(w: dict) -> dict:
    return {
        "id": w["id"], "title": w["title"], "year": w["year"], "citations": w["citations"],
        "doi": norm_doi(w.get("doi")), "pmid": None, "arxiv_id": None, "abstract": "", "tldr": "",
        "authors": w["authors"], "url": w["url"], "field": w.get("field"), "sources": ["openalex"],
    }


def _from_other(p: dict) -> dict:
    return {
        "id": None, "title": p["title"], "year": p["year"], "citations": p["citations"],
        "doi": norm_doi(p.get("doi")), "pmid": p.get("pmid"), "arxiv_id": p.get("arxiv_id"),
        "abstract": p["abstract"], "tldr": p["tldr"], "authors": p["authors"],
        "url": p["url"], "field": None, "sources": [p["source"]],
    }


def _merge(into: dict, other: dict) -> None:
    """Fold another copy of the same paper into the first one: keep what it lacks."""
    for field in ("abstract", "tldr", "doi", "pmid", "arxiv_id", "year", "field"):
        if not into.get(field) and other.get(field):
            into[field] = other[field]
    if (other.get("citations") or 0) > (into.get("citations") or 0):
        into["citations"] = other["citations"]
    if not any(a.get("id") for a in into["authors"]) and any(a.get("id") for a in other["authors"]):
        into["authors"] = other["authors"]
    into["sources"] += [s for s in other["sources"] if s not in into["sources"]]


# OpenAlex labels a paper by what it is applied to ("Pharmacology" for a graph neural network
# for drug discovery), so the field alone misses the method. These words in the query catch it.
PREPRINT_HINTS = re.compile(
    r"machine learning|deep learning|neural|transformer|language model|\bllms?\b|reinforcement|"
    r"\balgorithm|artificial intelligence|\bai\b|quantum|optimi[sz]ation|robot|computational|statistic",
    re.I,
)


def _pick_sources(oa_results: list[dict], query: str = "") -> list[str]:
    """Extra sources worth asking. A source is added when ANY of the papers OpenAlex found is in
    its field, not only the most common one, so a search that crosses fields reaches both."""
    fields = {w["field"] for w in oa_results if w.get("field")}
    extra = ["semantic_scholar"]
    if fields & BIOMEDICAL:
        extra.append("pubmed")
    if fields & PREPRINT or PREPRINT_HINTS.search(query):
        extra.append("arxiv")
    return extra


def search_papers(query: str, from_year: int | None = None, limit: int = 8, max_results: int = 15, sources: list[str] | None = None) -> dict:
    """Search every fitting source, merge, and attach OpenAlex author ids.

    sources: force a list of extra sources (any of semantic_scholar, pubmed, arxiv);
    by default they are chosen from the field of the OpenAlex results.
    Returns {"papers": [...], "sources_searched": [...], "errors": {source: message}}."""
    errors: dict[str, str] = {}
    papers: dict[str, dict] = {}

    try:
        oa_results = openalex.search_works(query, from_year, limit)
    except Exception as e:
        oa_results = []
        errors["openalex"] = f"{type(e).__name__}: {e}"
    searched = [] if "openalex" in errors else ["openalex"]
    for w in oa_results:
        p = _from_openalex(w)
        if _key(p) in papers:
            _merge(papers[_key(p)], p)
        else:
            papers[_key(p)] = p

    names = sources if sources is not None else _pick_sources(oa_results, query)
    pool = ThreadPoolExecutor(max_workers=max(1, len(names)))  # each source has its own rate limiter
    futures = {name: pool.submit(SOURCES[name], query, from_year, limit) for name in names}
    wait(futures.values(), timeout=SOURCE_DEADLINE)
    pool.shutdown(wait=False, cancel_futures=True)
    for name in names:
        future = futures[name]
        if not future.done():
            errors[name] = f"timed out after {SOURCE_DEADLINE} s, skipped"
            continue
        try:
            found = future.result()
            searched.append(name)
        except Exception as e:  # one source down must not sink the search
            errors[name] = f"{type(e).__name__}: {str(e)[:200]}"
            continue
        for raw in found:
            p = _from_other(raw)
            if _key(p) in papers:
                _merge(papers[_key(p)], p)
            else:
                papers[_key(p)] = p

    # Look every paper OpenAlex has not already given us up in OpenAlex (a free lookup)
    by_id = {p["id"]: p for p in papers.values() if p["id"]}
    for key, p in list(papers.items()):
        if p["id"]:
            continue
        try:
            hit = openalex.resolve(doi=p["doi"], pmid=p["pmid"], arxiv_id=p["arxiv_id"])
        except Exception:
            hit = None
        if not hit:
            continue  # kept as it is: authors have names but no ids
        if hit["id"] in by_id:  # the same paper OpenAlex returned under another key
            _merge(by_id[hit["id"]], p)
            del papers[key]
            continue
        p.update(id=hit["id"], authors=hit["authors"], url=hit["url"], field=hit.get("field"))
        p["citations"] = hit["citations"] if hit["citations"] is not None else p["citations"]
        p["doi"] = p["doi"] or norm_doi(hit.get("doi"))
        by_id[hit["id"]] = p

    # Found by several sources first, then most cited
    ranked = sorted(papers.values(), key=lambda p: (-len(p["sources"]), -(p["citations"] or 0)))
    return {"papers": ranked[:max_results], "sources_searched": searched, "errors": errors}


def source_summary(ok: dict[str, int], failed: dict[str, int], last_error: dict[str, str]) -> tuple[str, str]:
    """Per-run source status for the Writer. A source that failed in one search but worked in the others
    is reported as both, with counts, instead of as 'failed'."""
    searched = ", ".join(sorted(name for name, n in ok.items() if n)) or "none"
    lines = []
    for name in sorted(failed):
        total = ok.get(name, 0) + failed[name]
        lines.append(f"{name}: failed in {failed[name]} of {total} searches (last error: {last_error.get(name, 'unknown')[:120]})")
    return searched, "; ".join(lines) or "none"


def get_paper(paper_id: str) -> dict:
    """One paper in full: abstract, authors, the ids of the papers it cites.

    OpenAlex often has no abstract, so a missing one is filled from Semantic Scholar."""
    work = openalex.get_work(paper_id)
    if not work["abstract"] and work.get("doi"):
        try:
            other = semantic_scholar.get_paper("DOI:" + norm_doi(work["doi"]))
        except Exception:
            other = None
        if other:
            work["abstract"] = other["abstract"]
            work["tldr"] = other["tldr"]
    return work


def similar_papers(paper_id: str, limit: int = 8) -> list[dict]:
    """Papers Semantic Scholar finds similar, each looked up in OpenAlex for author ids."""
    doi = norm_doi(openalex.get_work(paper_id).get("doi"))
    if not doi:
        return []
    papers = []
    for raw in semantic_scholar.get_similar("DOI:" + doi, limit):
        p = _from_other(raw)
        hit = openalex.resolve(doi=p["doi"], pmid=p["pmid"], arxiv_id=p["arxiv_id"])
        if hit:
            p.update(id=hit["id"], authors=hit["authors"], url=hit["url"], field=hit.get("field"))
            p["citations"] = hit["citations"]
        papers.append(p)
    return papers


def compact(paper: dict, abstract_chars: int = 300) -> dict:
    """The short version the model reads: no author ids, abstracts cut short."""
    return {
        "id": paper["id"] or paper["pmid"] or paper["arxiv_id"],
        "title": paper["title"],
        "year": paper["year"],
        "citations": paper["citations"],
        "authors": [a["name"] for a in paper["authors"][:3]] + (["et al."] if len(paper["authors"]) > 3 else []),
        "summary": (paper.get("tldr") or paper["abstract"])[:abstract_chars],
        "found_in": paper["sources"],
    }


if __name__ == "__main__":
    # Smoke test: uv run papers.py "graph neural networks for drug discovery"
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "graph neural networks for drug discovery"
    result = search_papers(query, limit=5)
    print("searched:", result["sources_searched"])
    print("errors:  ", result["errors"] or "none")
    for p in result["papers"]:
        ids = sum(1 for a in p["authors"] if a.get("id"))
        print(f"\n{p['year']}  {p['citations']}  {p['title'][:80]}")
        print(f"   found in {p['sources']} | OpenAlex id: {p['id']} | authors with OpenAlex ids: {ids}/{len(p['authors'])}")
    assert result["papers"], "no papers at all"
    print("\nok")
