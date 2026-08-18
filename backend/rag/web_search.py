"""Dependency-free web search for the agent's ``tool`` route.

Uses the DuckDuckGo HTML endpoint (no API key, no SDK). The scraper is small
and defensive: any network/parse failure degrades to ``[]`` so the agent can
still answer gracefully ("web search unavailable") instead of crashing.
"""
import urllib.parse
import urllib.request
import re

from langchain_core.documents import Document

_SEARCH_ENDPOINT = "https://html.duckduckgo.com/html/"
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_TIMEOUT_SECS = 10
_MAX_RESULTS = 5


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_results(html: str, max_results: int = _MAX_RESULTS) -> list[Document]:
    """Extract {href, title, snippet} triples from DuckDuckGo's result markup."""
    docs: list[Document] = []
    # Each result is a <div class="result ..."> block.
    blocks = re.split(r'class="result\b', html)[1:]
    for block in blocks:
        title_m = re.search(r'class="result__a"[^>]*>(.*?)</a>', block, re.S)
        snip_m = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', block, re.S)
        href_m = re.search(r'class="result__a"\s+href="([^"]+)"', block)
        if not title_m:
            continue
        title = _clean(title_m.group(1))
        snippet = _clean(snip_m.group(1)) if snip_m else ""
        href = ""
        if href_m:
            # DuckDuckGo wraps real URLs in a redirect param.
            raw = href_m.group(1)
            q = urllib.parse.parse_qs(urllib.parse.urlparse(raw).query)
            href = q.get("uddg", [raw])[0]
        content = f"{title}\n{snippet}".strip()
        if content:
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "note_id": href or f"web://{urllib.parse.quote(title[:40])}",
                        "path": href,
                        "title": title,
                        "workspace": "web",
                        "source_type": "web",
                    },
                )
            )
    return docs[:max_results]


def web_search(query: str, max_results: int = _MAX_RESULTS) -> list[Document]:
    """Return top web results for ``query`` as Documents.

    Returns ``[]`` on any network/parse error so the agent loop degrades
    gracefully rather than raising. ``max_results`` is scoped to this call
    (no module-level mutation — avoids races when called concurrently)."""
    params = urllib.parse.urlencode({"q": query})
    req = urllib.request.Request(
        f"{_SEARCH_ENDPOINT}?{params}",
        headers={"User-Agent": _UA, "Accept-Language": "en-US,en;q=0.9"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_TIMEOUT_SECS) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception:
        return []
    return _parse_results(html, max_results)


def web_search_available() -> bool:
    """Cheap connectivity probe so callers can skip the search when offline."""
    try:
        req = urllib.request.Request(
            _SEARCH_ENDPOINT,
            headers={"User-Agent": _UA},
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=3):
            return True
    except Exception:
        return False