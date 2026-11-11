import asyncio
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from agents import Agent, ModelSettings, RunContextWrapper, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.litellm_model import LitellmModel
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import experts
import openalex
import papers
from context_panel import AgentContexts, PanelHooks, add_context_panel

# OPENALEX_API_KEY, S2_API_KEY, NCBI_API_KEY and VERTEXAI_PROJECT come from the .env file next to this one.
load_dotenv(Path(__file__).parent / ".env")

# Traces upload to OpenAI by default, which needs an OpenAI key. We're on Gemini.
set_tracing_disabled(True)

# LiteLLM routes the call to Gemini. api_key is unused: Vertex AI uses your gcloud credentials.
# If "gemini-3.5-flash" is not available in your project, use the flash-lite string for both.
MODEL = LitellmModel(model="vertex_ai/gemini-3.5-flash", api_key="unused")
# Researchers only search, read and take notes: the cheaper model is enough.
RESEARCHER_MODEL = LitellmModel(model="vertex_ai/gemini-3.5-flash-lite", api_key="unused")
# A stalled model call would hang the page: give up after 90 s, and retry twice.
SETTINGS = ModelSettings(extra_args={"vertex_location": "global", "timeout": 90, "num_retries": 2})

# The Lead gets this many researchers in all. Prompts ask for restraint; code enforces it.
MAX_RESEARCHERS = 6


MAX_RESULT_CHARS = 1500  # a tool result longer than this is cut in the /chat tool_calls, to keep the JSON small


class CountingHooks(PanelHooks):
    """PanelHooks that also (1) counts, for each agent, how many papers it opened with get_paper and
    (2) records every tool call of every agent as {agent, name, args, result}, for /chat's tool_calls."""

    def __init__(self, contexts, emit):
        super().__init__(contexts, emit)
        self.opened: dict[str, int] = {}
        self.calls: list[dict] = []  # in the order the calls started
        self._by_id: dict[str, dict] = {}

    async def on_tool_start(self, context, agent, tool):
        await super().on_tool_start(context, agent, tool)
        if tool.name == "get_paper":
            self.opened[agent.name] = self.opened.get(agent.name, 0) + 1
        raw = getattr(context, "tool_arguments", None)
        try:
            args = json.loads(raw) if raw else {}
        except (TypeError, ValueError):
            args = {"raw": str(raw)}
        call = {"agent": agent.name, "name": tool.name, "args": args, "result": None}
        self.calls.append(call)
        call_id = getattr(context, "tool_call_id", None)
        if call_id:
            self._by_id[call_id] = call

    async def on_tool_end(self, context, agent, tool, result):
        await super().on_tool_end(context, agent, tool, result)
        call = self._by_id.get(getattr(context, "tool_call_id", None))
        if call is None:  # no call id: the oldest unfinished call of this agent and tool
            call = next((c for c in self.calls if c["agent"] == agent.name and c["name"] == tool.name and c["result"] is None), None)
        if call is not None:
            text = str(result)
            call["result"] = text if len(text) <= MAX_RESULT_CHARS else text[:MAX_RESULT_CHARS] + f"... ({len(text)} characters in all)"


@dataclass
class Research:
    """One run's state, shared by every agent and tool in it. The model never sees it."""

    idea: str
    hooks: PanelHooks
    researchers: int = 0  # how many researchers have started, to number them
    findings: list[str] = field(default_factory=list)  # what each researcher reported, for the Writer
    # Every paper any tool returned, by id. The expert ranking reads authors from here, so the
    # model never has to name a person: it can only report papers that were really found.
    works: dict[str, dict] = field(default_factory=dict)
    source_ok: dict[str, int] = field(default_factory=dict)  # searches each database answered
    source_failed: dict[str, int] = field(default_factory=dict)  # searches each database failed
    source_errors: dict[str, str] = field(default_factory=dict)  # the last error message of each


# --- Helpers for the tools ---

OPENALEX_ID = re.compile(r"W\d+")


def _remember(research: Research, paper: dict) -> None:
    """Store a paper in the run's state, merging with what an earlier tool call already stored."""
    key = paper.get("id") or paper.get("pmid") or paper.get("arxiv_id") or paper["title"]
    research.works.setdefault(key, {}).update({k: v for k, v in paper.items() if v not in (None, "", [])})


def _for_model(paper: dict) -> dict:
    """What the model reads about a paper: short, with the link it must cite."""
    out = papers.compact(paper)
    out["url"] = paper.get("url")
    out["doi"] = paper.get("doi")
    return out


def _error(message: str) -> str:
    # The model cannot see an exception. Return something it can reason about.
    return json.dumps({"error": message})


NOT_OPENALEX = "Only papers with an OpenAlex id (like W3216638646) can be opened. For the others, use the summary from the search."


# --- Tools ---


@function_tool
async def search_papers(ctx: RunContextWrapper[Research], query: str, from_year: int = 0) -> str:
    """Search research papers across OpenAlex, Semantic Scholar, and PubMed or arXiv when the topic fits.
    Returns titles, years, citation counts, authors and a short summary. Each result has an id and a url.

    Args:
        query: A short English search: a few key terms, not a sentence. For example 'graph neural networks drug synergy'.
        from_year: Only papers from this year on, e.g. 2020. Use 0 for any year.
    """
    try:
        result = await asyncio.to_thread(papers.search_papers, query, from_year or None)
    except Exception as e:
        return _error(f"Search failed: {type(e).__name__}: {e}")
    for paper in result["papers"]:
        _remember(ctx.context, paper)
    for name in result["sources_searched"]:
        ctx.context.source_ok[name] = ctx.context.source_ok.get(name, 0) + 1
    for name, message in result["errors"].items():
        ctx.context.source_failed[name] = ctx.context.source_failed.get(name, 0) + 1
        ctx.context.source_errors[name] = message
    return json.dumps(
        {
            "sources_searched": result["sources_searched"],
            "source_errors": result["errors"],
            "papers": [_for_model(p) for p in result["papers"]],
            "hint": "Summaries only. Call get_paper to read an abstract, get_references or get_citing to follow the citation graph.",
        },
        ensure_ascii=False,
    )


@function_tool
async def get_paper(ctx: RunContextWrapper[Research], paper_id: str) -> str:
    """Read one paper: its abstract, authors and how many papers it cites.

    Args:
        paper_id: An OpenAlex id from a search result, like 'W3216638646'.
    """
    if not OPENALEX_ID.fullmatch(paper_id):
        return _error(NOT_OPENALEX)
    try:
        work = await asyncio.to_thread(papers.get_paper, paper_id)
    except Exception as e:
        return _error(f"Could not read {paper_id}: {type(e).__name__}: {e}")
    _remember(ctx.context, work)
    names = [a["name"] for a in work["authors"]]
    return json.dumps(
        {
            "id": work["id"],
            "title": work["title"],
            "year": work["year"],
            "citations": work["citations"],
            "url": work["url"],
            "doi": work.get("doi"),
            "authors": names if len(names) <= 6 else names[:5] + names[-1:],
            "abstract": work["abstract"][:1500] or "(OpenAlex and Semantic Scholar have no abstract for this paper)",
            "tldr": work.get("tldr", ""),
            "references_count": len(work["reference_ids"]),
        },
        ensure_ascii=False,
    )


async def _citation_list(ctx: RunContextWrapper[Research], fetch, paper_id: str) -> str:
    if not OPENALEX_ID.fullmatch(paper_id):
        return _error(NOT_OPENALEX)
    try:
        found = await asyncio.to_thread(fetch, paper_id)
    except Exception as e:
        return _error(f"Lookup failed for {paper_id}: {type(e).__name__}: {e}")
    records = [papers._from_openalex(w) for w in found]
    for record in records:
        _remember(ctx.context, record)
    return json.dumps({"papers": [_for_model(r) for r in records]}, ensure_ascii=False)


@function_tool
async def get_references(ctx: RunContextWrapper[Research], paper_id: str) -> str:
    """The papers this paper cites, most cited first. These are usually the foundations of the paper's topic.

    Args:
        paper_id: An OpenAlex id, like 'W3216638646'.
    """
    return await _citation_list(ctx, openalex.get_references, paper_id)


@function_tool
async def get_citing(ctx: RunContextWrapper[Research], paper_id: str) -> str:
    """The papers that cite this paper, most cited first. These are usually newer work building on it.

    Args:
        paper_id: An OpenAlex id, like 'W3216638646'.
    """
    return await _citation_list(ctx, openalex.get_citing, paper_id)


# --- The researcher: one angle of the idea, by searching and walking the citation graph ---

researcher = Agent(
    name="Researcher",
    instructions="""\
You research one angle of a user's research idea, using paper databases.

1. Search with search_papers. Try 2 or 3 differently worded queries: short, English, key terms and
   not sentences. Use from_year when recent work matters.
2. Open the 3 to 5 most relevant papers with get_paper. You must open at least 3 before you reply:
   a search summary alone is not enough to say what a paper did. The summaries are enough for the rest.
3. Follow the citation graph from your best one or two papers: get_references finds the
   foundations, get_citing finds newer work that builds on it.
4. Make at most 10 tool calls in all.
5. Reply with at most 8 bullets and nothing else. Each bullet starts with exactly one tag in square
   brackets: [very close], [related method], [related problem] or [background]. Then it says what the
   paper did or found, and ends with its title, year and link, e.g.
   - [very close] Used an LLM to ... ([Paper title](https://openalex.org/W123), 2021)
   Use only what the tools returned: never fill in details from memory.
   End with a last bullet saying what you looked for and did not find.""",
    model=RESEARCHER_MODEL,
    model_settings=SETTINGS,
    tools=[search_papers, get_paper, get_references, get_citing],
)


@function_tool
async def research(ctx: RunContextWrapper[Research], task: str, seed_query: str) -> str:
    """Send a researcher to investigate one angle of the idea by searching papers and following citations.
    Returns its findings, each with a link to the paper.

    Args:
        task: One specific thing to find out, e.g. 'Find existing work that applies graph neural networks to predicting drug synergy'.
        seed_query: A short English search to start from, e.g. 'graph neural network drug synergy'.
    """
    if ctx.context.researchers >= MAX_RESEARCHERS:
        return "No researchers left. Reply DONE."
    # What researcher.as_tool() does: run the researcher in a fresh context and return its answer.
    # Written out so each run gets a numbered copy, and its own column in the context panel.
    ctx.context.researchers += 1
    agent = researcher.clone(name=f"Researcher {ctx.context.researchers}")
    prompt = f"The user's idea: {ctx.context.idea}\nYour task: {task}\nStart by searching: {seed_query}"
    result = await Runner.run(agent, prompt, context=ctx.context, hooks=ctx.context.hooks, max_turns=14)
    text = str(result.final_output)
    if getattr(ctx.context.hooks, "opened", {}).get(agent.name, 0) == 0:
        # Counted by the hooks, not reported by the model: it cannot hide having skipped the reading
        text += "\n\n(Note: this researcher opened no paper with get_paper. Its bullets rest on search-result summaries only.)"
    ctx.context.findings.append(f"Task: {task}\n\n{text}")
    return text


# --- The lead: plans, fans out to researchers, and checks for gaps ---

lead = Agent(
    name="Lead",
    instructions="""\
You lead a research team. The user gives you a rough research idea. The team finds the existing
research that is related to it and how new the idea looks. The idea may not be in English: search in English.

1. Call search_papers once or twice to see what the field looks like.
2. Split the idea into 2 to 4 independent angles. For example: the closest existing work to the whole
   idea; the method or technique on its own; the problem or application on its own; how such work is
   evaluated and where it falls short. Call research once per angle, all in the same turn, so the
   researchers work in parallel. Every angle must come from the user's own words: do not add concepts,
   settings or constraints the user did not mention. Do not put a technique, architecture, training
   method or constraint in a task or a seed_query unless the user named it, not even as "something to
   look for": a short, broad idea gets broad angles. Give each a task of one short sentence (it labels
   the angle in the report) and a short English seed_query.
3. Read the findings. If an important angle is still uncovered, call research again for just that.
   Do this at most twice.
4. When the idea is covered, reply DONE and nothing else: a writer writes the report.""",
    model=MODEL,
    model_settings=SETTINGS,
    tools=[search_papers, research],
)

# --- The writer: one call, after the research ends ---

writer = Agent(
    name="Writer",
    instructions="""\
Write a short report in markdown for a researcher who has an idea. Use only the findings and the
candidate list you're given. Write in the language the idea is written in.

- Start with a short answer of two or three sentences: which part of the idea existing work already
  covers, which part we did not find work on, and how close the closest paper is. Say how little was
  searched (a few queries, abstracts only). Never call the idea new, novel, distinct or unexplored: a
  search that finds nothing does not show that nothing exists. Give counts as "papers cited in the
  findings", never as "relevant papers".
- "Closest existing work": the 3 to 6 papers most like the idea, one short paragraph each: what it
  did, and how it differs from the idea.
- "Already well covered" and "Gaps": what the findings show is done, and what they do not show.
  Write gaps only about what the user's own idea says: never add concepts the user did not mention
  (if the idea says "large companies", do not turn it into "bureaucracy" or "hierarchy"; if it says
  "LLM with image prediction", do not invent a gap about diffusion heads, other architectures or
  reinforcement learning). Before you write a gap, check that every
  noun phrase in it appears in the idea's own words (or is a plain synonym). If a gap needs a word the
  idea does not contain ("clinical", "longitudinal", "Fortune 500", "attrition", "multinational",
  "syntax errors"), drop it. The same goes for the short answer: describe the idea only in its own words.
  Good gaps compare what the idea combines with what the findings show: the two parts together, one
  part in the idea's own setting, or the idea's own outcome. Say the idea is too broad to name gaps
  only when it does not say what is done and for what purpose; then write one line and name the details
  that would let a search say it. A clear question is not too broad. Merge gaps
  that say the same thing. Phrase every gap as "we did not find".
- "Who to talk to": from the candidate list only, pick 3 to 5 people. For each, give the name, the
  affiliation as listed, why they fit (which of their papers, and whether they were first or last
  author), and one concrete question to ask them. Each candidate shows the angle of the search it
  was found through: pick people from different angles where you can, and say which side of the idea
  each covers. If the idea mixes fields (a method from one, a problem from another), cover both.
  Give each person their own reason, taken from their papers; say they cover both sides only if their listed papers show both.
  Never use gendered pronouns (he, she, his, her): the data does not say. Use the person's name or "they".
  Never name a person who is not in the candidate list. Never give an email address. If the list
  is empty, say that no candidates could be ranked.
- "Limits of this search": name the databases searched. Say if a database failed, and in how many of its searches (it may have worked in the others). Say that only titles,
  abstracts and summaries were read, never full papers. If a finding carries a note that its researcher
  opened no paper, say which angle that was and that it rests on search summaries only. Say that the candidates are the authors of
  the closest papers found, not a measure of who is the leading expert, and that affiliations may be
  out of date.
- Keep the links: end every sentence that states a fact about a paper with its link from the findings,
  written as a markdown link with the paper's short title, like [Short title](https://openalex.org/W123).
  Never write a bare url in square brackets.
- Write plain, specific sentences. Don't claim more significance than the findings do: no
  "profoundly", "pivotal", "paving the way", or "it is worth noting".""",
    model=MODEL,
    model_settings=SETTINGS,
)


# --- The clarifier: the first step of /chat. A vague idea gets one question back, not a search ---

clarifier = Agent(
    name="Clarifier",
    instructions="""\
You are the first step of a tool that finds research related to a researcher's idea and says who to talk to.
Decide whether the message is specific enough to search for.

It is specific enough if it says what is done to or with what, and for what purpose or in what setting
(for example "using large language models to find errors in clinical trial protocols"), even if it is
rough or short. Reply READY and nothing else.

It is too vague if it only puts topics side by side without saying what is to be studied, built or
compared ("LLM with image prediction", "machine learning for healthcare", "blockchain and education"),
if it is only a broad field or a single word ("AI", "health", "climate"), a greeting, a question that
is not a research idea, or so open that it could mean several very different studies.
Then reply with ONE short question, in the language of the message, that would help most: what does the
user want to do, find out or build with it? Offer two or three example directions inside the question.
Do not ask about anything a search could find out.
The message may already contain the user's answers to your earlier questions: judge the whole text. If
the user says they do not know, offer three concrete directions to choose from.
Never answer the idea yourself.""",
    model=MODEL,
    model_settings=SETTINGS,
)


async def clarify(idea: str, hooks: PanelHooks) -> str | None:
    """None if the idea is specific enough to research, else the question to ask the user."""
    result = await Runner.run(clarifier, f"Message: {idea}", hooks=hooks, max_turns=2)
    text = str(result.final_output).strip()
    return None if not text or text.upper().startswith("READY") else text


async def intake(session_id: str, message: str, hooks: PanelHooks) -> tuple[str | None, str]:
    """Decide what to do with a user message. Returns (question, idea): a question to ask the user first,
    or None and the idea to research. An answer is checked again, so "I don't know" gets a better question,
    not a search; after MAX_QUESTIONS the search runs anyway."""
    prior, asked = pending_ideas.pop(session_id, (None, 0))
    idea = message if prior is None else f"{prior}\n(The user's answer to a clarifying question: {message})"
    if asked < MAX_QUESTIONS:
        question = await clarify(idea, hooks)
        if question:
            pending_ideas[session_id] = (idea, asked + 1)
            return question, idea
    return None, idea


LINK = re.compile(r"\[([^\]]*)\]\(https?://openalex\.org/(W\d+)\)")


def verify_links(report: str, known: set[str]) -> tuple[str, list[str]]:
    """Every OpenAlex link in the report must be a paper the tools really returned. The Writer is told to
    copy links from the findings, but a model can still mistype or invent an id. A link that matches no
    found paper is turned into plain text and reported, so no reader follows a link to a paper we never saw."""
    removed: list[str] = []

    def fix(match):
        title, wid = match.group(1), match.group(2)
        if wid in known:
            return match.group(0)
        removed.append(wid)
        return f"{title} (link removed: not a paper found in this search)"

    return LINK.sub(fix, report), removed


async def deep_research(idea: str, hooks: PanelHooks) -> tuple[str, Research]:
    research_run = Research(idea=idea, hooks=hooks)

    # 1. Research: plan, fan out, check for gaps
    await Runner.run(lead, f"Idea: {idea}", context=research_run, hooks=hooks, max_turns=12)

    # 2. Rank experts in plain Python from the papers the Researchers cited. No model names a person.
    ranking = experts.rank_experts(research_run.works, research_run.findings)

    # 3. Write: one call, outside the loop. The Writer sees the idea, the findings and the ranked candidates.
    findings = "\n\n---\n\n".join(research_run.findings) or "(the researchers returned no findings)"
    searched, failed = papers.source_summary(research_run.source_ok, research_run.source_failed, research_run.source_errors)
    report = await Runner.run(
        writer,
        f"The user's idea: {idea}\n\nDatabases searched: {searched}\nDatabases that failed: {failed}\n\n"
        f"Findings:\n\n{findings}\n\nCandidates to talk to (ranked by code):\n\n{experts.for_writer(ranking)}",
        hooks=hooks,
    )
    table = experts.evidence_table(ranking)  # built by code from the same data, so every name can be checked
    text, removed = verify_links(str(report.final_output), set(research_run.works))
    if removed:
        text += f"\n\n_Check: {len(removed)} link(s) in the draft pointed to papers that were not found in this search and were removed._"
    return (text + ("\n\n" + table if table else "")), research_run


# --- FastAPI App ---

HERE = Path(__file__).parent

# run_id -> every agent's context, in order. In-memory, single process. The context panel reads it.
sessions: dict[str, AgentContexts] = {}

# session_id -> (the idea so far, how many questions we asked), waiting for the user's answer. In memory, one process.
pending_ideas: dict[str, tuple[str, int]] = {}
MAX_QUESTIONS = 2  # after this many questions we research whatever we have, and the report says the idea is broad

app = FastAPI()
add_context_panel(app, sessions)


@app.get("/")
def index():
    # no-cache: browsers otherwise keep serving an old page after a redeploy
    return FileResponse(HERE / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/run")
async def run(idea: str, session_id: str | None = None):
    """Research one idea and stream what happens as Server-Sent Events.
    A vague idea gets a "clarify" event with a question; the page sends the answer with the same session_id."""
    run_id = str(uuid.uuid4())
    sessions[run_id] = AgentContexts()
    queue: asyncio.Queue = asyncio.Queue()
    hooks = CountingHooks(sessions[run_id], queue.put_nowait)

    async def work():
        started = time.monotonic()
        try:
            sid = session_id or str(uuid.uuid4())
            question, full_idea = await intake(sid, idea, hooks)
            if question:
                queue.put_nowait({"type": "clarify", "question": question, "session_id": sid})
                return
            report, research_run = await deep_research(full_idea, hooks)
            queue.put_nowait({"type": "done", "markdown": report, "papers": len(research_run.works), "seconds": round(time.monotonic() - started)})
        except Exception as e:
            # Auth, quota, a model that won't stop: show it on the page, not as a dropped stream.
            queue.put_nowait({"type": "error", "message": f"{type(e).__name__}: {str(e)[:500]}"})

    task = asyncio.create_task(work())

    async def events():
        try:
            yield f"data: {json.dumps({'type': 'start', 'run_id': run_id})}\n\n"
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event, default=str)}\n\n"
                if event["type"] in ("done", "error", "clarify"):
                    break
        finally:
            task.cancel()  # the page closed the stream: stop spending tokens

    return StreamingResponse(events(), media_type="text/event-stream")


# --- /chat: the same research, as one JSON call (the shape from Project 1, plus an agent on each tool call) ---

CHAT_TIMEOUT = 270  # seconds; a caller that waits 5 minutes gets an answer instead of a dropped request


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    response: str
    session_id: str
    tool_calls: list[dict]  # each: {"agent", "name", "args", "result"}, for the Lead and every Researcher


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    message = request.message.strip()
    run_id = str(uuid.uuid4())
    sessions[run_id] = AgentContexts()
    hooks = CountingHooks(sessions[run_id], lambda event: None)

    def reply(text: str) -> ChatResponse:
        return ChatResponse(response=text, session_id=session_id, tool_calls=hooks.calls)

    if not message:
        return reply("Please describe your research idea in a sentence or two.")

    try:
        question, idea = await asyncio.wait_for(intake(session_id, message, hooks), timeout=60)
        if question:
            return reply(question)
        report, _ = await asyncio.wait_for(deep_research(idea, hooks), timeout=CHAT_TIMEOUT)
        return reply(report)
    except asyncio.TimeoutError:
        return reply("The search took too long and was stopped. Try a shorter, more specific idea.")
    except Exception as e:
        return reply(f"Something went wrong: {type(e).__name__}: {str(e)[:300]}")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
