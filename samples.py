"""Run the three sample ideas for the README, two ways, and save the answers to read side by side.

    uv run samples.py baseline                 the same model with NO tools and NO system prompt
    uv run samples.py agent                    the agent, through POST /chat on http://127.0.0.1:8000
    uv run samples.py agent https://your-app   ... or on the deployed app

Writes baseline_answers.md or agent_answers.md. The agent takes 1 to 3 minutes per idea.
"""

import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

SAMPLES = [
    "Can large language models catch errors in clinical trial protocols?",
    "Graph neural networks for predicting drug-drug interactions in patients taking five or more medications",
    "Does remote work increase loneliness among employees at large companies?",
]
MODEL = "vertex_ai/gemini-3.5-flash"  # the same model the Lead and the Writer use


def baseline(idea: str) -> str:
    import litellm

    # Only the idea: no system prompt, no tools
    response = litellm.completion(
        model=MODEL, messages=[{"role": "user", "content": idea}],
        vertex_location="global", vertex_project=os.environ.get("VERTEXAI_PROJECT"),
    )
    return response.choices[0].message.content


def agent(idea: str, url: str) -> str:
    r = httpx.post(url.rstrip("/") + "/chat", json={"message": idea}, timeout=330)
    r.raise_for_status()
    data = r.json()
    calls = data["tool_calls"]
    by_agent: dict[str, int] = {}
    for c in calls:
        by_agent[c["agent"]] = by_agent.get(c["agent"], 0) + 1
    return data["response"] + f"\n\n---\n_{len(calls)} tool calls: {json.dumps(by_agent)}_"


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode not in ("baseline", "agent"):
        sys.exit(__doc__)
    url = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8000"
    out = []
    for i, idea in enumerate(SAMPLES, 1):
        print(f"[{i}/{len(SAMPLES)}] {idea}", flush=True)
        try:
            answer = baseline(idea) if mode == "baseline" else agent(idea, url)
        except Exception as e:
            answer = f"(failed: {type(e).__name__}: {str(e)[:300]})"
        out.append(f"## Idea {i}: {idea}\n\n{answer}\n")
    path = Path(__file__).parent / f"{mode}_answers.md"
    path.write_text("\n".join(out))
    print("saved", path)
