"""One polite HTTP client per source: disk cache, shared rate limit, retries.

Every source has its own rules (arXiv: one request every 3 seconds on one connection;
Semantic Scholar: 1 per second with a key; PubMed: 3 per second, 10 with a key). The
Researchers run in parallel, so the limit must be shared by every thread: that is what
the lock is for. Cached answers cost nothing and never touch the network.
"""

import hashlib
import json
import threading
import time
from pathlib import Path

import httpx

CACHE_ROOT = Path(__file__).parent / ".cache"
RETRY_STATUS = (429, 500, 502, 503, 504)


def check(response: httpx.Response) -> None:
    """Like raise_for_status, but the message shows the server's reply and leaves out the query string.
    The default message prints the whole URL, and OpenAlex and NCBI keys travel in it: they would end
    up in tool results, the report and the context panel."""
    if response.is_error:
        shown = str(response.url).split("?")[0]
        raise httpx.HTTPStatusError(
            f"{response.status_code} for {shown}: {response.text[:300].strip()}", request=response.request, response=response
        )


class Client:
    def __init__(self, name: str, min_interval: float, serialize: bool = False, timeout: float = 30, attempts: int = 4,
                 user_agent: str = "deep-research-papers (course project)"):
        """min_interval: seconds between two calls. serialize: hold the lock during the whole
        request, so only one is in flight at a time (arXiv asks for a single connection)."""
        self.min_interval = min_interval
        self.timeout = timeout  # seconds to wait for one response
        self.attempts = attempts  # tries before giving up: worst case is about timeout x attempts
        self.serialize = serialize
        self.headers = {"User-Agent": user_agent}
        self.cache_dir = CACHE_ROOT / name
        self._lock = threading.Lock()
        self._last = 0.0

    def _pace(self) -> None:  # the caller holds the lock
        wait = self._last + self.min_interval - time.time()
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def _send(self, url: str, params: dict, headers: dict) -> httpx.Response:
        return httpx.get(url, params=params, headers=headers, timeout=self.timeout, follow_redirects=True)

    def get(self, url: str, params: dict | None = None, secret: dict | None = None, headers: dict | None = None, text: bool = False):
        """GET url. Returns parsed JSON, or the raw text with text=True.
        secret: params (an API key) sent but kept out of the cache key and the cache file."""
        params = params or {}
        key = hashlib.sha1((url + json.dumps(params, sort_keys=True)).encode()).hexdigest()
        cache_file = self.cache_dir / (key + (".txt" if text else ".json"))
        if cache_file.exists():
            body = cache_file.read_text()
            return body if text else json.loads(body)

        sent_params = {**params, **(secret or {})}
        sent_headers = {**self.headers, **(headers or {})}
        for attempt in range(self.attempts):
            try:
                if self.serialize:
                    with self._lock:
                        self._pace()
                        response = self._send(url, sent_params, sent_headers)
                else:
                    with self._lock:
                        self._pace()
                    response = self._send(url, sent_params, sent_headers)
            except httpx.TransportError:  # timeout, dropped connection
                if attempt == self.attempts - 1:
                    raise
                time.sleep(2 ** (attempt + 1))
                continue
            if response.status_code in RETRY_STATUS and attempt < self.attempts - 1:
                try:
                    wait = min(float(response.headers.get("Retry-After", "")), 10)
                except ValueError:
                    wait = 2 ** (attempt + 1)  # 2 s, 4 s, 8 s
                time.sleep(wait)
                continue
            check(response)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(response.text if text else json.dumps(response.json()))
            return response.text if text else response.json()
        raise RuntimeError("unreachable")
