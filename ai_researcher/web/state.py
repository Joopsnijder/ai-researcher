"""Shared state management for research sessions."""

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ResearchStatus(Enum):
    """Status of a research session."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ResearchSession:
    """Represents a single research session."""

    id: str
    question: str
    template: Optional[str] = None
    status: ResearchStatus = ResearchStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    report_path: Optional[str] = None
    error: Optional[str] = None

    # Progress tracking
    iteration_count: int = 0
    recursion_limit: int = 200
    searches_count: int = 0
    cache_hits: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    current_status: str = "Waiting..."
    todos: list = field(default_factory=list)
    recent_searches: list = field(default_factory=list)

    # Event queue for SSE
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)

    # Background thread
    _thread: Optional[threading.Thread] = None

    def to_dict(self) -> dict:
        """Convert session to dictionary for JSON serialization."""
        elapsed = None
        if self.started_at:
            end = self.completed_at or datetime.now()
            elapsed = (end - self.started_at).total_seconds()

        return {
            "id": self.id,
            "question": self.question,
            "template": self.template,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "elapsed_seconds": elapsed,
            "report_path": self.report_path,
            "error": self.error,
            "iteration_count": self.iteration_count,
            "recursion_limit": self.recursion_limit,
            "searches_count": self.searches_count,
            "cache_hits": self.cache_hits,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "current_status": self.current_status,
            "todos": self.todos,
            "recent_searches": self.recent_searches[-5:],  # Last 5 searches
        }


class ResearchStateManager:
    """Manages all research sessions."""

    def __init__(self):
        self._sessions: dict[str, ResearchSession] = {}
        self._lock = threading.Lock()

    def create_session(
        self, question: str, template: Optional[str] = None
    ) -> ResearchSession:
        """Create a new research session."""
        session_id = str(uuid.uuid4())[:8]
        session = ResearchSession(
            id=session_id,
            question=question,
            template=template,
        )
        with self._lock:
            self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> Optional[ResearchSession]:
        """Get a research session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def list_sessions(self) -> list[ResearchSession]:
        """List all research sessions."""
        with self._lock:
            return list(self._sessions.values())

    def cleanup(self):
        """Cleanup all sessions on shutdown."""
        with self._lock:
            for session in self._sessions.values():
                if session.status == ResearchStatus.RUNNING:
                    session.status = ResearchStatus.CANCELLED
            self._sessions.clear()


# Global state instance
research_state = ResearchStateManager()
