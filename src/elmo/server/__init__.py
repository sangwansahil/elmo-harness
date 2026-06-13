"""elmo daemon — FastAPI + single-page UI under localhost:7777."""

from .app import create_app, serve

__all__ = ["create_app", "serve"]
