"""YouTube transcript ingestion via ``youtube-transcript-api``.

Transcript segments are merged into ~CHUNK_SIZE chunks that preserve accurate
``timestamp_start``/``timestamp_end`` metadata, so answers can link back to the
moment in the video where the claim appears.
"""
import re
import urllib.request
from typing import Any

from config import settings
from rag import embedder as _embedder
from rag.vectorstore import stable_chunk_id, vectorstore
from youtube_transcript_api import YouTubeTranscriptApi


def _extract_video_id(url: str) -> str | None:
    patterns = [
        r"youtu\.be/([0-9A-Za-z_-]{11})",
        r"v=([0-9A-Za-z_-]{11})",
        r"embed/([0-9A-Za-z_-]{11})",
        r"/shorts/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _fetch_video_title(video_id: str) -> str:
    url = f"https://www.youtube.com/watch?v={video_id}"
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read().decode("utf-8", errors="ignore")
            match = re.search(r"<title>(.*?)</title>", html)
            if match:
                return match.group(1).replace(" - YouTube", "").strip()
    except Exception:  # noqa: BLE001, S110
        pass
    return f"YouTube Video ({video_id})"


def _fetch_transcript(video_id: str) -> list[dict[str, Any]]:
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)
    try:
        transcript = transcript_list.find_manually_created_transcript(["en"])
    except Exception:  # noqa: BLE001
        try:
            transcript = transcript_list.find_generated_transcript(["en"])
        except Exception:  # noqa: BLE001
            transcript = next(iter(transcript_list))
    return list(transcript.fetch())


def chunk_transcript(
    transcript: list[dict[str, Any]], video_id: str, url: str, title: str
) -> list[dict]:
    """Merge transcript segments into chunks with accurate timestamps.

    Uses explicit chunk-size constants from settings instead of reaching into
    the text splitter's private attributes (the old Conversational behaviour).
    """
    if not transcript:
        return []

    chunk_size, chunk_overlap = settings.CHUNK_SIZE, settings.CHUNK_OVERLAP
    chunks: list[dict] = []

    current_content: list[str] = []
    current_start = transcript[0]["start"]
    current_length = 0

    for i, entry in enumerate(transcript):
        current_content.append(entry["text"])
        current_length += len(entry["text"]) + 1  # +1 for the joining space

        if current_length >= chunk_size or i == len(transcript) - 1:
            content = " ".join(current_content)
            end = entry["start"] + entry.get("duration", 0)
            chunks.append(
                {
                    "id": stable_chunk_id(video_id, len(chunks)),
                    "text": content,
                    "metadata": {
                        "note_id": video_id,
                        "path": url,
                        "workspace": "default",  # caller re-scopes per workspace
                        "title": title,
                        "source_type": "youtube",
                        "timestamp_start": current_start,
                        "timestamp_end": end,
                        "chunk_index": len(chunks),
                    },
                }
            )

            if i < len(transcript) - 1:
                # Carry the tail segments into the next chunk as overlap, and
                # map the next chunk's start timestamp to the first kept segment.
                overlap_length, overlap_entries = 0, []
                for seg in reversed(current_content):
                    overlap_length += len(seg) + 1
                    overlap_entries.insert(0, seg)
                    if overlap_length >= chunk_overlap:
                        break
                current_content = overlap_entries
                current_length = overlap_length
                entry_index = i - len(overlap_entries) + 1
                current_start = transcript[entry_index]["start"]
            else:
                current_content = []
                current_length = 0

    return chunks


def chunk_youtube(url: str, workspace: str = "default") -> list[dict]:
    """Fetch + chunk a YouTube transcript. Returns chunk dicts (no embeddings)."""
    video_id = _extract_video_id(url)
    if not video_id:
        raise ValueError(f"Could not extract video ID from URL: {url}")

    title = _fetch_video_title(video_id)
    transcript = _fetch_transcript(video_id)
    chunks = chunk_transcript(transcript, video_id, url, title)
    for chunk in chunks:
        chunk["metadata"]["workspace"] = workspace
    return chunks


def ingest_youtube(url: str, workspace: str = "default") -> int:
    """Index a YouTube transcript into the vector store. Returns chunk count."""
    chunks = chunk_youtube(url, workspace)
    if not chunks:
        return 0
    video_id = chunks[0]["metadata"]["note_id"]
    vectorstore.delete_note_chunks(video_id)
    texts = [c["text"] for c in chunks]
    embeddings = _embedder.embed_documents(texts)
    embedded = [{**c, "embedding": emb} for c, emb in zip(chunks, embeddings)]
    return vectorstore.upsert_chunks(embedded, workspace=workspace)
