"""Rank the researchers a user should talk to, in plain Python. No LLM touches a name.

Input: every paper the tools returned in a run (research.works) and the Researchers' findings.
Only papers the Researchers cited in their findings count. Otherwise background papers that came
back in a search (a famous optimizer paper, say) would put unrelated people at the top.

Authors are matched by OpenAlex author id, never by name, so one person is never split in two
and two people with the same name are never merged. A cited paper that OpenAlex does not know
has no author ids: it is counted in "papers_without_ids" and skipped.

Score per paper an author appears on:
    position weight  x  (recency + a small citation bonus)
    position weight: first or last author 1.5, middle 0.6, plus 0.25 if corresponding author
    recency: 1.0 for this year, 0.1 less per year, never below 0.3
    citation bonus: 0.1 x log10(1 + citations)
Each paper is also multiplied by its closeness tag: very close 2.0, related method or problem 1.0, background 0.4.
Papers add up, so someone on several of the cited papers outranks someone on one.

Who is listed (so the list covers the idea, not just its most crowded corner):
    Each Researcher's finding is one angle of the idea. Every angle is ranked on its own papers, and
    the list takes each angle's best candidates in turn (per_angle rounds), then fills the rest of the
    places from the overall ranking.
    At most one person per paper: someone is skipped when every paper they are first or last author on
    (or, for a middle author, every cited paper) already brought in a person. This stops the first and
    last author of one paper, usually from the same lab, from filling two places.
    The final list is ordered by the overall score.
"""

import math
import re
from datetime import date

OPENALEX_LINK = re.compile(r"openalex\.org/(W\d+)")
MAX_PAPERS_SHOWN = 4  # evidence papers listed per person
ANGLE_SHOWN = 60  # characters of an angle's name in the table
# Each Researcher bullet starts with one of these tags. A closer paper counts for more.
CLOSENESS = {"very close": 2.0, "related method": 1.0, "related problem": 1.0, "background": 0.4}
CLOSENESS_TAG = re.compile(r"\[(very close|related method|related problem|background)\]", re.I)


def cited_ids(findings: list[str]) -> list[str]:
    """OpenAlex paper ids linked in the findings, in order, without repeats."""
    seen: dict[str, None] = {}
    for text in findings:
        for match in OPENALEX_LINK.findall(text):
            seen.setdefault(match)
    return list(seen)


def closeness_by_paper(findings: list[str]) -> dict[str, str]:
    """{OpenAlex id: closeness tag} from the bullets. A paper in several bullets keeps its closest tag.
    A bullet with no tag gets no entry: the paper then counts as 'related' (weight 1.0)."""
    best: dict[str, str] = {}
    for text in findings:
        for line in text.splitlines():
            tag = CLOSENESS_TAG.search(line)
            if not tag:
                continue
            label = tag.group(1).lower()
            for wid in OPENALEX_LINK.findall(line):
                if wid not in best or CLOSENESS[label] > CLOSENESS[best[wid]]:
                    best[wid] = label
    return best


def angle_label(finding: str) -> str:
    """The task the Lead gave a Researcher: findings start with a 'Task: ...' line. '' if they do not."""
    first = next((line.strip() for line in finding.splitlines() if line.strip()), "")
    return first.removeprefix("Task:").strip() if first.startswith("Task:") else ""


def _paper_weight(paper: dict, this_year: int) -> float:
    year = paper.get("year") or this_year - 10
    recency = max(0.3, 1 - 0.1 * max(0, this_year - year))
    return recency + 0.1 * math.log10(1 + (paper.get("citations") or 0))


def _score(works: dict[str, dict], ids: list[str], closeness: dict[str, str], this_year: int):
    """Score the authors of these cited papers. Returns (people, used, without_ids, unknown)."""
    people: dict[str, dict] = {}
    used = without_ids = 0
    unknown = []
    for wid in ids:
        paper = works.get(wid)
        if not paper:
            unknown.append(wid)  # cited by a researcher but never returned by a tool
            continue
        authors = [a for a in paper.get("authors", []) if a.get("id")]
        if not authors:
            without_ids += 1
            continue
        used += 1
        label = closeness.get(wid)
        paper_weight = _paper_weight(paper, this_year) * CLOSENESS.get(label, 1.0)
        for a in authors:
            senior = a.get("position") in ("first", "last")
            weight = (1.5 if senior else 0.6) + (0.25 if a.get("corresponding") else 0)
            person = people.setdefault(a["id"], {
                "id": a["id"], "name": a.get("name"), "score": 0.0, "papers": [],
                "first_or_last": 0, "institution": None, "_institution_year": -1,
                "anchors": set(),  # cited papers they are first or last author on
                "cited": set(),  # all cited papers they are on
            })
            person["score"] += weight * paper_weight
            person["first_or_last"] += int(senior)
            person["cited"].add(wid)
            if senior:
                person["anchors"].add(wid)
            year = paper.get("year") or 0
            person["papers"].append({
                "id": wid, "title": paper.get("title"), "year": paper.get("year"), "url": paper.get("url"),
                "position": a.get("position"), "closeness": label,
            })
            # The affiliation printed on the newest paper is the best guess at where they are now
            if a.get("institutions") and year >= person["_institution_year"]:
                person["institution"] = a["institutions"][0]
                person["_institution_year"] = year
    return people, used, without_ids, unknown


def _order(people: dict[str, dict]) -> list[dict]:
    return sorted(people.values(), key=lambda p: (-p["score"], -len(p["papers"]), p["name"] or ""))


def rank_experts(works: dict[str, dict], findings: list[str], top_n: int = 8, per_angle: int = 2,
                 one_per_paper: bool = True, this_year: int | None = None) -> dict:
    """Pick and rank the authors to talk to. Returns {"experts": [...], "angles", "papers_cited",
    "papers_used", "papers_without_ids", "papers_unknown"} so the report can say what the ranking rests on."""
    this_year = this_year or date.today().year
    ids = cited_ids(findings)
    people, used, without_ids, unknown = _score(works, ids, closeness_by_paper(findings), this_year)

    # Each finding is one angle of the idea: rank its authors on that angle's papers alone
    angles: list[tuple[str, list[str]]] = []
    for finding in findings:
        angle_people, *_ = _score(works, cited_ids([finding]), closeness_by_paper([finding]), this_year)
        angles.append((angle_label(finding), [p["id"] for p in _order(angle_people)]))

    chosen: dict[str, int | None] = {}  # person id -> index of the angle they were picked for
    covered: set[str] = set()  # papers that already brought in a person

    def fits(pid: str) -> bool:
        if pid in chosen:
            return False
        if not one_per_paper:
            return True
        return bool((people[pid]["anchors"] or people[pid]["cited"]) - covered)

    def take(pid: str, angle: int | None) -> None:
        chosen[pid] = angle
        covered.update(people[pid]["anchors"] or people[pid]["cited"])

    for _round in range(per_angle):
        for i, (_name, ranking) in enumerate(angles):
            if len(chosen) >= top_n:
                break
            pick = next((pid for pid in ranking if fits(pid)), None)
            if pick:
                take(pick, i)
    for p in _order(people):  # fill the remaining places from the overall ranking
        if len(chosen) >= top_n:
            break
        if fits(p["id"]):
            home = next((i for i, (_name, ranking) in enumerate(angles) if p["id"] in ranking), None)
            take(p["id"], home)

    ranked = _order({pid: people[pid] for pid in chosen})
    experts = []
    for rank, p in enumerate(ranked, 1):
        papers = sorted(p["papers"], key=lambda x: -(x["year"] or 0))
        angle = chosen[p["id"]]
        experts.append({
            "rank": rank,
            "id": p["id"],
            "name": p["name"],
            "profile": f"https://openalex.org/{p['id']}",
            "institution": p["institution"],
            "angle": angles[angle][0] if angle is not None else "",
            "papers_found": len(papers),
            "first_or_last_author_on": p["first_or_last"],
            "latest_year": papers[0]["year"],
            "score": round(p["score"], 2),
            "papers": papers[:MAX_PAPERS_SHOWN],
        })
    return {
        "experts": experts,
        "angles": len(angles),
        "papers_cited": len(ids),
        "papers_used": used,
        "papers_without_ids": without_ids,
        "papers_unknown": len(unknown),
    }


def for_writer(result: dict) -> str:
    """The ranking as text for the Writer: only these people may be named."""
    if not result["experts"]:
        return "(No candidates: none of the cited papers had authors with OpenAlex ids.)"
    lines = []
    for e in result["experts"]:
        papers = "; ".join(f"[{p['title']}]({p['url']}) ({p['year']}, {p['position'] or 'author'} author, {p['closeness'] or 'untagged'})" for p in e["papers"])
        lines.append(
            f"{e['rank']}. {e['name']} | latest listed affiliation: {e['institution'] or 'unknown'} | "
            f"found via the angle: {e['angle'] or 'n/a'} | "
            f"{e['papers_found']} cited paper(s), first or last author on {e['first_or_last_author_on']} | "
            f"profile: {e['profile']} | papers: {papers}"
        )
    lines.append(
        f"\nThe ranking rests on {result['papers_used']} of the {result['papers_cited']} papers cited in the findings. "
        "Each angle of the search contributed its best candidates, and at most one person is listed per paper."
    )
    return "\n".join(lines)


def _cell(text: str, limit: int) -> str:
    text = (text or "").replace("|", "/").strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def evidence_table(result: dict) -> str:
    """A markdown table built by code from the same data, appended after the Writer's report."""
    if not result["experts"]:
        return ""
    rows = [
        "| # | Researcher | Latest listed affiliation | Found via angle | Cited papers | First/last author on | Latest paper |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in result["experts"]:
        rows.append(
            f"| {e['rank']} | [{e['name']}]({e['profile']}) | {e['institution'] or 'unknown'} | "
            f"{_cell(e['angle'], ANGLE_SHOWN) or 'n/a'} | {e['papers_found']} | {e['first_or_last_author_on']} | {e['latest_year']} |"
        )
    note = (
        f"Chosen and ranked by code, not by a model, from {result['papers_used']} of the {result['papers_cited']} papers cited above "
        f"(authors matched by OpenAlex id). Each angle of the search contributed its best candidates, and at most one person is listed per paper. "
    )
    if result["papers_without_ids"] or result["papers_unknown"]:
        note += f"{result['papers_without_ids'] + result['papers_unknown']} cited paper(s) had no author ids and were skipped. "
    note += "This shows who wrote the closest papers we found, not who is the leading expert. Check each profile for current affiliation before contacting anyone."
    return "## Evidence behind the expert ranking\n\n" + "\n".join(rows) + "\n\n" + note


if __name__ == "__main__":
    # Self-check: uv run experts.py
    def author(i, position="middle", corresponding=False, inst="MIT"):
        return {"id": f"A{i}", "name": f"Person {i}", "position": position, "corresponding": corresponding, "institutions": [inst]}

    def paper(wid, year, authors, citations=10):
        return {"id": wid, "title": wid, "year": year, "citations": citations, "url": f"https://openalex.org/{wid}", "authors": authors}

    works = {
        "W1": {"id": "W1", "title": "Close paper", "year": 2025, "citations": 100, "url": "https://openalex.org/W1",
               "authors": [author(1, "first"), author(2, "middle"), author(3, "last", True, "Stanford")]},
        "W2": {"id": "W2", "title": "Related paper", "year": 2023, "citations": 10, "url": "https://openalex.org/W2",
               "authors": [author(3, "first", inst="Harvard"), author(4, "last")]},
        "W3": {"id": "W3", "title": "Background: Adam", "year": 2014, "citations": 90000, "url": "https://openalex.org/W3",
               "authors": [author(9, "first")]},                       # returned by a tool, never cited
        "W4": {"id": "W4", "title": "No ids", "year": 2024, "citations": 0, "url": "https://openalex.org/W4",
               "authors": [{"id": None, "name": "Nobody"}]},
    }
    findings = [
        "- Did X ([Close paper](https://openalex.org/W1), 2025)\n- Did Y ([Related paper](https://openalex.org/W2), 2023)",
        "- Again ([Close paper](https://openalex.org/W1), 2025)\n- ([No ids](https://openalex.org/W4), 2024)\n- ([Ghost](https://openalex.org/W99))",
    ]
    assert cited_ids(findings) == ["W1", "W2", "W4", "W99"]
    r = rank_experts(works, findings, one_per_paper=False, this_year=2026)   # scoring alone, without the one-per-paper rule
    names = [e["name"] for e in r["experts"]]
    assert names[0] == "Person 3", names                               # on both papers, senior on both
    assert "Person 9" not in names                                     # background paper never cited
    assert names.index("Person 2") > names.index("Person 4")          # middle author ranks below a last author
    assert r["papers_cited"] == 4 and r["papers_used"] == 2 and r["papers_without_ids"] == 1 and r["papers_unknown"] == 1
    top = r["experts"][0]
    assert top["institution"] == "Stanford" and top["papers_found"] == 2 and top["first_or_last_author_on"] == 2  # newest paper's affiliation
    assert top["profile"] == "https://openalex.org/A3"
    # closeness: a "very close" paper outranks a "background" one even if the background one is newer and cited more
    works2 = {
        "W10": {"id": "W10", "title": "Closest", "year": 2023, "citations": 5, "url": "u10", "authors": [author(10, "first")]},
        "W11": {"id": "W11", "title": "Survey", "year": 2026, "citations": 900, "url": "u11", "authors": [author(11, "first")]},
    }
    tagged = ["- [very close] Did it ([Closest](https://openalex.org/W10), 2023)\n- [background] Overview ([Survey](https://openalex.org/W11), 2026)"]
    assert closeness_by_paper(tagged) == {"W10": "very close", "W11": "background"}
    assert [e["name"] for e in rank_experts(works2, tagged, this_year=2026)["experts"]][0] == "Person 10"
    untagged = ["- ([Closest](https://openalex.org/W10), 2023)\n- ([Survey](https://openalex.org/W11), 2026)"]
    assert [e["name"] for e in rank_experts(works2, untagged, this_year=2026)["experts"]][0] == "Person 11"  # no tags: old behaviour
    assert rank_experts({}, [], this_year=2026)["experts"] == []
    assert "Person 3" in for_writer(r) and "Evidence behind" in evidence_table(r)
    assert evidence_table({"experts": []}) == ""

    # Angles: the crowded machine-learning side must not push out the one clinical candidate
    works3 = {
        "W101": paper("W101", 2026, [author(21, "first"), author(22, "last")]),          # first and last author of ONE paper
        "W102": paper("W102", 2026, [author(23, "first"), author(24, "middle")]),
        "W103": paper("W103", 2022, [author(31, "first")]),
    }
    f_ml = "Task: ML side | graph methods\n\n- [very close] a ([W101](https://openalex.org/W101))\n- [very close] b ([W102](https://openalex.org/W102))"
    f_clin = "Task: Clinical side\n\n- [related problem] c ([W103](https://openalex.org/W103))"
    assert angle_label(f_clin) == "Clinical side" and angle_label("- no task line") == "" and angle_label("") == ""
    overall = [e["name"] for e in rank_experts(works3, [f_ml, f_clin], top_n=2, per_angle=0, one_per_paper=False, this_year=2026)["experts"]]
    assert overall == ["Person 21", "Person 22"], overall                          # without angles: the ML side takes both places
    two = rank_experts(works3, [f_ml, f_clin], top_n=2, per_angle=1, one_per_paper=False, this_year=2026)
    assert {e["name"] for e in two["experts"]} == {"Person 21", "Person 31"}, two  # with angles: one from each side
    assert {e["name"]: e["angle"] for e in two["experts"]} == {"Person 21": "ML side | graph methods", "Person 31": "Clinical side"}
    every = {e["name"] for e in rank_experts(works3, [f_ml, f_clin], one_per_paper=False, this_year=2026)["experts"]}
    assert every == {"Person 21", "Person 22", "Person 23", "Person 24", "Person 31"}
    one = rank_experts(works3, [f_ml, f_clin], this_year=2026)                      # default: at most one person per paper
    assert {e["name"] for e in one["experts"]} == {"Person 21", "Person 23", "Person 31"}, one   # 22 shares M1 with 21, 24 shares M2 with 23
    assert one["angles"] == 2
    assert "found via the angle: Clinical side" in for_writer(one)
    table = evidence_table(one)
    assert "Clinical side" in table and "ML side / graph methods" in table and "|" not in "ML side / graph methods"
    print(table)
    print("\nok")
