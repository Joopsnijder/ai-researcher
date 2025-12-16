"""FastAPI routes for AI Researcher web interface."""

import asyncio
import os
from datetime import datetime

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from ..templates import get_template_info
from ..config import RESEARCH_FOLDER
from .state import research_state, ResearchSession, ResearchStatus
from .sse import format_sse_event


router = APIRouter()


class StartResearchRequest(BaseModel):
    """Request body for starting research."""

    question: str
    template: str | None = None
    recursion_limit: int = 200


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    templates = request.app.state.templates
    template_list = get_template_info()
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "templates": template_list,
        },
    )


@router.post("/research/start")
async def start_research(
    request: Request, body: StartResearchRequest, background_tasks: BackgroundTasks
):
    """Start a new research session."""
    # Create session
    session = research_state.create_session(
        question=body.question,
        template=body.template,
    )

    # Start research in background thread
    background_tasks.add_task(
        run_research_background,
        session,
        body.recursion_limit,
    )

    return {
        "research_id": session.id,
        "status": "started",
        "stream_url": f"/research/{session.id}/stream",
    }


@router.get("/research/{research_id}/status")
async def get_research_status(research_id: str):
    """Get the current status of a research session."""
    session = research_state.get_session(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")
    return session.to_dict()


@router.get("/research/{research_id}/stream")
async def stream_research_events(research_id: str):
    """Stream research events via Server-Sent Events."""
    session = research_state.get_session(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    async def event_generator():
        """Generate SSE events from the session queue."""
        # Send initial status
        yield format_sse_event("status", session.to_dict())

        while True:
            try:
                # Wait for events with timeout
                event = await asyncio.wait_for(session.event_queue.get(), timeout=30.0)

                if event is None:  # End signal
                    yield format_sse_event("complete", session.to_dict())
                    break

                event_type = event.get("type", "update")
                event_data = event.get("data", {})
                yield format_sse_event(event_type, event_data)

            except asyncio.TimeoutError:
                # Send keepalive
                yield format_sse_event(
                    "keepalive", {"timestamp": datetime.now().isoformat()}
                )

            except Exception as e:
                yield format_sse_event("error", {"message": str(e)})
                break

            # Check if session is done
            if session.status in (
                ResearchStatus.COMPLETED,
                ResearchStatus.FAILED,
                ResearchStatus.CANCELLED,
            ):
                yield format_sse_event("complete", session.to_dict())
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/research/{research_id}/stop")
async def stop_research(research_id: str):
    """Stop a running research session."""
    session = research_state.get_session(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    if session.status != ResearchStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Research is not running")

    session.status = ResearchStatus.CANCELLED
    session.completed_at = datetime.now()

    # Signal end of stream
    await session.event_queue.put(None)

    return {"status": "cancelled"}


@router.get("/research/{research_id}/report")
async def download_report(research_id: str):
    """Download the research report as Markdown."""
    session = research_state.get_session(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    if not session.report_path or not os.path.exists(session.report_path):
        raise HTTPException(status_code=404, detail="Report not found")

    return FileResponse(
        session.report_path,
        media_type="text/markdown",
        filename=os.path.basename(session.report_path),
    )


@router.get("/templates")
async def get_templates():
    """Get list of available templates."""
    return get_template_info()


@router.get("/research/{research_id}", response_class=HTMLResponse)
async def research_page(request: Request, research_id: str):
    """Render the research progress page."""
    session = research_state.get_session(research_id)
    if not session:
        raise HTTPException(status_code=404, detail="Research session not found")

    templates = request.app.state.templates
    return templates.TemplateResponse(
        "research.html",
        {
            "request": request,
            "session": session,
        },
    )


def run_research_background(session: ResearchSession, recursion_limit: int):
    """Run research in a background thread."""
    import asyncio
    from ..tracking import AgentTracker
    from ..search import HybridSearchTool, SearchStatusDisplay
    from ..runners import run_research

    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        session.status = ResearchStatus.RUNNING
        session.started_at = datetime.now()
        session.recursion_limit = recursion_limit

        # Create callback-enabled tracker
        tracker = AgentTracker()
        tracker.iteration_count = 0
        tracker.recursion_limit = recursion_limit

        # Create callback to push updates to SSE queue
        def on_update(update_type: str, data: dict):
            """Callback for research updates."""
            # Update session state
            session.iteration_count = tracker.iteration_count
            session.searches_count = tracker.searches_count
            session.cache_hits = getattr(tracker, "cache_hits", 0)
            session.total_input_tokens = tracker.total_input_tokens
            session.total_output_tokens = tracker.total_output_tokens
            session.current_status = getattr(tracker, "current_status", "Processing...")
            session.todos = getattr(tracker, "current_todos", []) or []

            # Push event to queue
            try:
                loop.call_soon_threadsafe(
                    session.event_queue.put_nowait,
                    {"type": update_type, "data": session.to_dict()},
                )
            except Exception:
                pass  # Queue might be full or closed

        # Attach callback to tracker
        tracker.on_update = on_update

        # Create search tool with display
        search_tool = HybridSearchTool(provider="multi-search")
        search_display = SearchStatusDisplay()

        # Override search display to capture searches
        original_add = search_display.add_search

        def capture_search(*args, **kwargs):
            original_add(*args, **kwargs)
            session.recent_searches = list(search_display.recent_searches)
            on_update("search", {"searches": session.recent_searches})

        search_display.add_search = capture_search

        # Run research
        run_research(
            question=session.question,
            recursion_limit=recursion_limit,
            tracker=tracker,
            search_tool=search_tool,
            search_display=search_display,
            template=session.template,
        )

        # Find the report
        import glob

        report_files = glob.glob(os.path.join(RESEARCH_FOLDER, "*.md"))
        if report_files:
            # Get most recent
            session.report_path = max(report_files, key=os.path.getmtime)

        session.status = ResearchStatus.COMPLETED
        session.completed_at = datetime.now()

    except Exception as e:
        session.status = ResearchStatus.FAILED
        session.error = str(e)
        session.completed_at = datetime.now()

    finally:
        # Signal end of stream
        try:
            loop.call_soon_threadsafe(session.event_queue.put_nowait, None)
        except Exception:
            pass
        loop.close()
