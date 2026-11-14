"""FastAPI application: one process, one SQLite file, one inference module.

    uvicorn backend.app.main:app --reload        # from the project root
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from backend.app.api.routes import router  # noqa: E402
from backend.app.core.config import Settings, get_settings  # noqa: E402
from backend.app.models.db import make_session_factory  # noqa: E402
from backend.app.services.scoring import ModelService  # noqa: E402


def create_app(settings: Settings | None = None, model_service=None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="Electricity Theft Detection API",
        description="Risk scores for smart-meter consumption. SUSPICIOUS means investigation recommended, "
                    "not proof of theft.",
        version="1.0.0",
    )
    app.state.settings = settings
    app.state.session_factory = make_session_factory(settings.database_url)
    app.state.model_service = model_service or ModelService(settings.model_metadata_path)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list,
                       allow_methods=["*"], allow_headers=["*"])
    app.include_router(router)
    return app


app = create_app()
