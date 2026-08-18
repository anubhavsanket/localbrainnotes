"""Pydantic request/response schemas shared by the API layer."""
from pydantic import BaseModel


class NoteWrite(BaseModel):
    """Payload for creating/updating a note file in the vault."""
    path: str
    content: str
    workspace: str = "default"


class QueryRequest(BaseModel):
    question: str
    workspace: str = "default"


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_seconds: float


class WorkspaceResponse(BaseModel):
    workspaces: list[str]
    count: int
