# Research Idea Explorer

**Path:** Deep Research · **Live app:** `FILL IN AFTER DEPLOY` (no setup or API key needed)

## Who it is for and what it does

Research Idea Explorer is for **researchers and graduate students who have a rough research idea** and want to know, before they start, what has already been done and whom to ask. You type an idea in a sentence or two (any language). A **Lead** agent splits it into 2 to 4 angles, **Researcher** agents search the paper databases and follow citations in parallel, and a **Writer** reports:

- the closest existing papers and how each differs from your idea,
- what is already well covered, and what we **did not find** (never "this is new"),
- **who to talk to**: authors of the closest papers, chosen and ranked by plain Python code from OpenAlex author ids (no model ever names a person), with an evidence table so every name can be checked.

If the idea is too broad ("AI", "LLM with image prediction"), it asks up to two clarifying questions first.

## Data sources

| Source | Used for | Notes |
|---|---|---|
| [OpenAlex](https://openalex.org) | Search, abstracts, citation graph (references and citing papers), author ids and affiliations | Backbone: every paper from any source is looked up here again, so authors carry OpenAlex ids |
| [Semantic Scholar](https://www.semanticscholar.org/product/api) | Search by meaning, TLDR summaries, abstracts OpenAlex lacks, similar papers | Always searched |
| [PubMed](https://pubmed.ncbi.nlm.nih.gov) (NCBI E-utilities) | Biomedical search | Searched when the topic is biomedical |
| [arXiv](https://arxiv.org) | Preprints in computer science, math, physics | Searched when the topic fits |

## Agents

| Agent | Role |
|---|---|
| Clarifier | First step: decides if the idea is specific enough; if not, asks one short question (max 2) |
| Lead | Splits the idea into angles, fans out one Researcher per angle in parallel (fresh contexts), reads the findings, goes back for gaps (max 2 more rounds, max 6 Researchers) |
| Researcher (several, parallel) | One angle each: `search_papers`, `get_paper`, `get_references`, `get_citing`; told to open at least 3 papers; code counts the `get_paper` calls and flags a Researcher that opened none |
| Writer | One call after the research: writes the report from the findings and the code-ranked candidate list |

Tools: `search_papers` (all sources, merged and deduplicated), `get_paper`, `get_references`, `get_citing`. Tools return errors as JSON so the model can recover; one source failing never fails a search.

## How to use it

Open the app, type your idea (or click an example), press **Find related research**, and wait 1 to 3 minutes. Answer the question if it asks one. The report links every paper it mentions. **Show what the agents did** opens a panel with every agent's context and tool calls.

API: `POST /chat` with `{"message": "...", "session_id": null}` returns `{"response", "session_id", "tool_calls"}` where each tool call is `{"agent", "name", "args", "result"}` for the Lead and every Researcher. Send the same `session_id` to answer a clarifying question.

## Sample queries

Each idea was run through the agent (`uv run samples.py agent`) and through the same model, Gemini 3.5 Flash, with no tools and no system prompt (`uv run samples.py baseline`). The full answers of both are in `agent_answers.md` and `baseline_answers.md` in this repo. Links are OpenAlex pages of papers the agent retrieved; every link in a report is checked by code against the papers the tools returned.

**The agent was run twice per idea, and the runs differ** (the model chooses its own queries). The expected answers below therefore state what the cited papers say and which run found which paper; the part both runs agree on is the one to rely on.

### 1. Can large language models catch errors in clinical trial protocols?

- **Expected answer:** Existing work has used an LLM to review clinical trial protocols against regulatory guidance: GPT-4o on the statistical-analysis and pharmacokinetics sections against FDA guidelines ([Protocol Assessments](https://openalex.org/W4415416600), found in both runs). Related work: LLMs and retrieval-augmented generation checking protocols for regulatory compliance ([RAG for regulatory compliance](https://openalex.org/W7130503162)), LLMs translating protocols into formal logic ([Ensemble Logic](https://openalex.org/W7170638918)), and LLMs detecting methodological errors in trial reports ([Methodological errors](https://openalex.org/W7224080690), found in run 1 only). Neither run found a study testing LLMs on catching errors across a whole protocol. Murali Krishna Ramanathan, last author of the first paper, is in both runs' lists of people to talk to.
- **Source:** OpenAlex (abstracts read by the agent); open the first link to confirm.
- **Agent's answer (run 2)**, which begins:

  > Existing work covers using large language models to assess clinical trial protocols for regulatory compliance, but we did not find work on using large language models to catch errors in clinical trial protocols. The closest paper is very close, evaluating how a large language model reviews clinical trial protocols against regulatory guidance [Large Language Models for Clinical Trial Protocol Assessments](https://openalex.org/W4415416600).

  It continues with four closest papers, what is well covered, one gap, four people to talk to with a question each, the limits (it reports that Semantic Scholar failed in 2 of 9 searches and that some angles rest on search summaries only), and an evidence table (22 of 22 cited papers matched to OpenAlex author ids). 28 tool calls.
- **Baseline (same model, no tools, no system prompt)**, which begins:

  > Yes, large language models (LLMs) can catch errors in clinical trial protocols, but their effectiveness depends heavily on the **type of error**, the **architecture of the LLM system** being used, and the presence of **human oversight**. Currently, LLMs are best viewed as highly efficient **co-pilots** or "second eyes" rather than independent auditors.

  It goes on with a general essay: what LLMs catch (internal inconsistencies, missing sections, schedule mismatches), what they miss, and how industry deploys them (RAG, private deployments, multi-agent review). It cites no paper and no author, so a reader cannot tell what has actually been tested or by whom.

### 2. Graph neural networks for predicting drug-drug interactions in patients taking five or more medications

- **Expected answer:** Graph and hypergraph neural networks for drug-drug interactions are well established: Decagon, a graph convolutional network for polypharmacy side effects ([Modeling polypharmacy side effects](https://openalex.org/W2786016794)), and a sparse hypergraph network for higher-order combinations ([SPARSE](https://openalex.org/W4225351203), found in both runs); run 1 also found a hypergraph model for high-order interactions ([HyHO-DDI](https://openalex.org/W7221509664)). Run 2 found a paper that proposes a conceptual framework with graph neural networks for patients taking five or more medications ([Graph Neural Networks for Drug-Drug Interaction Prediction](https://openalex.org/W7211911824)); run 1 missed it and reported that nothing on five or more medications was found. Neither run found a fully evaluated model built for that group.
- **Source:** OpenAlex (abstracts read by the agent); PubMed and arXiv were also searched.
- **Agent's answer (run 2)**, which begins:

  > Existing work extensively covers using graph neural networks to predict drug-drug interactions [Modeling polypharmacy side effects](https://openalex.org/W2786016794). However, we did not find work evaluating graph neural networks for predicting drug-drug interactions in patients taking five or more medications. The closest paper is a conceptual framework that proposes utilizing graph neural networks to predict drug-drug interactions in patients taking five or more medications [Graph Neural Networks for Drug-Drug Interaction Prediction](https://openalex.org/W7211911824).

  It continues with four closest papers, two gaps, four people (for example Jure Leskovec, last author of Decagon) with a question each, and an evidence table (17 of 17 cited papers matched). 24 tool calls.
- **Baseline (same model, no tools, no system prompt)**, which begins:

  > Using **Graph Neural Networks (GNNs)** to predict drug-drug interactions (DDIs) in patients taking five or more medications (a clinical state known as **polypharmacy** or **hyperpolypharmacy**) is one of the most promising and active frontiers in bioinformatics and AI-driven medicine.

  It goes on with a long tutorial on how GNNs could model polypharmacy. It names models ("MuzGNN", "HP-DDI") that a quick web search did not turn up under those names, and gives no link to check. Its combinatorics is off: it says five drugs give "26 possible multi-drug combinations (triplets, quadruplets, etc.)", but 26 counts the 10 pairs too (16 groups have three or more drugs). It calls the area a promising frontier without saying whether anyone has studied five or more drugs, which is the idea's own question.

### 3. Does remote work increase loneliness among employees at large companies?

- **Expected answer:** Yes, studies link more remote work with more loneliness: a two-wave survey of remote employees ([Lonely@Work@Home?](https://openalex.org/W4372342495), run 2); working remotely three or more days a week raised the odds of higher loneliness in a nationally representative U.S. sample ([Remote work and loneliness](https://openalex.org/W4415279844), run 1); remote work was positively related to loneliness in a representative Norwegian sample ([The pros and cons of remote work](https://openalex.org/W4307195080), run 1); newcomers in a large technology company working hybrid were lonelier than experienced employees ([Determinants of Work Loneliness](https://openalex.org/W4407180945), run 1). Both runs agree on the gap: no study found that compares remote-work loneliness by company size.
- **Source:** OpenAlex (abstracts read by the agent). The two runs cited different papers.
- **Agent's answer (run 2)**, which begins:

  > Existing work covers how remote work affects workplace loneliness among employees, with the closest studies directly surveying remote workers to evaluate their loneliness. However, we did not find research specifically examining whether this relationship is affected by working at large companies.

  It continues with three closest studies, two gaps, four people with a question each (for example, whether their survey recorded company size), the limits (Semantic Scholar failed in 1 of 5 searches; one angle rests on search summaries only), and an evidence table (18 of 18 cited papers matched). 20 tool calls.
- **Baseline (same model, no tools, no system prompt)**, which begins:

  > The short answer is **yes, remote work generally increases the risk of loneliness and isolation**, particularly for employees at large companies. However, the relationship is complex. Whether a remote worker feels lonely depends heavily on the company’s culture, the employee's life stage, and how the remote work is structured (e.g., fully remote vs. hybrid).

  It says "academic studies consistently show" this but names no study, sample or number. The large-company part (silos, resource groups, retreat budgets) is plausible reasoning with no source, so a reader cannot tell which parts are evidence and which are opinion.

## How it works

```
Browser ── GET /run?idea=  (streams progress)      Grader ── POST /chat
              │                                                  │
              └───────────────────┬──────────────────────────────┘
                                  ▼
                        Clarifier → question? ── yes → ask the user (max 2)
                                  │ no
                                  ▼
                      Lead ── research() ×N in parallel ──► Researcher 1..N (fresh contexts)
                        ▲                │                       │ search_papers / get_paper /
                        └─ findings ─────┘                       │ get_references / get_citing
                                  │
                      rank experts in Python (OpenAlex author ids)
                                  ▼
                      Writer → report → every OpenAlex link checked against the papers really found
```

### Design choices

- **No model names a person.** Candidates come from the authors of papers the Researchers actually cited. Score = author position (first/last count most) × recency and citations × how close the paper is to the idea; each search angle contributes its best candidates, at most one person per paper.
- **Links are checked by code.** Any report link to a paper that was not returned by a tool is removed.
- **Honest limits in every report:** which databases failed and how many searches, abstracts only, "we did not find" is not "does not exist".
- **Bounded cost and time:** max 6 Researchers, max 10 tool calls each, 35 s deadline per extra source, 270 s cap on `/chat`.

## Limitations

- Only titles, abstracts and summaries are read, never full papers.
- Results vary between runs (the model chooses the queries): for the remote-work idea two runs cited entirely different papers, and for the drug-interaction idea one run found a paper on patients taking five or more medications that the other missed, so its "we did not find" differed. A "we did not find" is a statement about this search, not about the literature.
- Authors and affiliations come from OpenAlex only and may be out of date. The "people to talk to" are authors of the closest papers, not a ranking of the leading experts.
- Strongest in medicine, biology and computer science; thinner for humanities and non-English work.
- Free API tiers rate-limit: a source can fail in some searches (the report says so).
- Sessions and the paper cache live in memory/disk of one instance; the app is deployed with a single instance.

## Run locally

```
cp .env.example .env      # VERTEXAI_PROJECT, OPENALEX_API_KEY, S2_API_KEY, NCBI_API_KEY
gcloud auth application-default login
uv run app.py             # http://127.0.0.1:8000
```
