"""Search biomedical papers on PubMed through NCBI's E-utilities.

Two calls per search: esearch finds the PubMed ids, efetch returns the records as XML
(title, abstract, authors, DOI). No key is needed, but NCBI_API_KEY in .env raises the
limit from 3 to 10 requests per second. NCBI_EMAIL identifies this tool to NCBI.

PubMed has no citation counts, and its author names have no ids: papers.py looks each
paper up in OpenAlex afterwards.
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv

from _http import Client

load_dotenv(Path(__file__).parent / ".env")

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
MAX_AUTHORS = 10

# 3 requests per second without a key, 10 with one (a little slower than the limit, for margin)
_client = Client("pubmed", min_interval=0.12 if os.environ.get("NCBI_API_KEY") else 0.35, timeout=20, attempts=3)


def _identity() -> tuple[dict, dict]:
    """(params that go in the cache key, params that stay out of it)."""
    public = {"tool": "deep-research-papers"}
    if os.environ.get("NCBI_EMAIL"):
        public["email"] = os.environ["NCBI_EMAIL"]
    secret = {"api_key": os.environ["NCBI_API_KEY"]} if os.environ.get("NCBI_API_KEY") else {}
    return public, secret


def _text(element) -> str:
    """All the text inside an element, including text inside tags like <i> or <sub>."""
    return "".join(element.itertext()).strip() if element is not None else ""


def parse_articles(xml_text: str) -> list[dict]:
    """PubMed XML -> one normalized record per article."""
    papers = []
    for article in ET.fromstring(xml_text).findall(".//PubmedArticle"):
        citation = article.find("MedlineCitation")
        body = citation.find("Article")
        if body is None:
            continue
        abstract = []
        for part in body.findall("Abstract/AbstractText"):
            label = part.get("Label")
            abstract.append(f"{label}: {_text(part)}" if label else _text(part))
        year = _text(body.find("Journal/JournalIssue/PubDate/Year")) or _text(body.find("ArticleDate/Year"))
        if not year:  # some records only have a free-text date like '2019 Jan-Feb'
            match = re.search(r"\d{4}", _text(body.find("Journal/JournalIssue/PubDate/MedlineDate")))
            year = match.group(0) if match else ""
        authors = []
        for au in body.findall("AuthorList/Author")[:MAX_AUTHORS]:
            name = f"{_text(au.find('ForeName'))} {_text(au.find('LastName'))}".strip() or _text(au.find("CollectiveName"))
            if name:
                authors.append({"id": None, "name": name})
        doi = next((_text(i) for i in article.findall("PubmedData/ArticleIdList/ArticleId") if i.get("IdType") == "doi"), None)
        pmid = _text(citation.find("PMID"))
        papers.append({
            "source": "pubmed",
            "source_id": pmid,
            "title": _text(body.find("ArticleTitle")).rstrip("."),
            "year": int(year) if year else None,
            "abstract": "\n".join(abstract),
            "tldr": "",
            "citations": None,
            "doi": doi,
            "pmid": pmid,
            "arxiv_id": None,
            "authors": authors,
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        })
    return papers


def search(query: str, from_year: int | None = None, limit: int = 10) -> list[dict]:
    """Papers matching a query, most relevant first."""
    public, secret = _identity()
    params = {"db": "pubmed", "term": query, "retmax": min(limit, 25), "retmode": "json", "sort": "relevance", **public}
    if from_year:
        params.update(datetype="pdat", mindate=str(from_year), maxdate="3000")
    ids = _client.get(f"{BASE}/esearch.fcgi", params, secret=secret)["esearchresult"].get("idlist", [])
    if not ids:
        return []
    fetch = {"db": "pubmed", "id": ",".join(ids), "retmode": "xml", **public}
    papers = parse_articles(_client.get(f"{BASE}/efetch.fcgi", fetch, secret=secret, text=True))
    order = {pmid: i for i, pmid in enumerate(ids)}  # efetch does not promise to keep the order
    return sorted(papers, key=lambda p: order.get(p["pmid"], len(order)))


if __name__ == "__main__":
    # Smoke test: uv run pubmed.py "CRISPR gene therapy sickle cell"
    import sys

    query = sys.argv[1] if len(sys.argv) > 1 else "CRISPR gene therapy sickle cell"
    print("API key:", "yes" if os.environ.get("NCBI_API_KEY") else "no (3 requests/second)")
    results = search(query, limit=5)
    assert results, "search returned nothing"
    for r in results:
        print(f"{r['year']}  PMID {r['pmid']}  {r['title'][:90]}")
        print("   authors:", [a["name"] for a in r["authors"][:3]], "| doi:", r["doi"])
        print("   abstract:", r["abstract"][:100] or "(none)")
    print("\nok")
