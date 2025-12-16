"""FastAPI application for AI Researcher web interface."""

from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .routes import router
from .state import research_state


# Get the directory containing this file
WEB_DIR = Path(__file__).parent
TEMPLATES_DIR = WEB_DIR / "templates"
STATIC_DIR = WEB_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    yield
    # Shutdown - cleanup any running research
    research_state.cleanup()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="AI Research Agent",
        description="Web interface for the AI Research Agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Mount static files
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Include routes
    app.include_router(router)

    # Store templates in app state for access in routes
    app.state.templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    return app


def run_server(host: str = "127.0.0.1", port: int = 8000, reload: bool = False):
    """Run the web server."""
    import uvicorn

    uvicorn.run(
        "ai_researcher.web.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


# Create app instance for uvicorn
app = create_app()
