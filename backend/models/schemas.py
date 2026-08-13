"""Pydantic request/response schemas shared by the API layer."""
from pydantic import BaseModel


class NoteCreate(BaseModel):
    """A single note body sent to the API (kept for endpoint compatibility)."""
    id: str
    content: str
    workspace: str = "default"


class NoteUpdate(BaseModel):
    content: str
    workspace: str = "default"


class QueryRequest(BaseModel):
    question: str
    workspace: str = "default"
    has_images: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_seconds: float


class WorkspaceResponse(BaseModel):
    workspaces: list[str]
    count: int
