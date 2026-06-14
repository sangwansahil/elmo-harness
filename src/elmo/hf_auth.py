"""Hugging Face auth + dataset-access plumbing.

Wraps huggingface_hub's token machinery and the public datasets metadata
endpoint so the wizard can hand-hold a first-time user through:

  1. "We need a Hugging Face read token" → paste here → we validate it
     and persist it via the HF standard location (~/.cache/huggingface/token).
  2. "This dataset is gated. Click here to request access" → after the
     user is approved, "check again" verifies and unlocks training.

No background polling. Everything is on-demand from a UI action. Lean.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass

HF_API = "https://huggingface.co/api"


@dataclass
class WhoAmI:
    configured: bool
    username: str = ""
    error: str = ""

    def asdict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetAccess:
    repo_id: str
    accessible: bool
    gated: str | bool = False        # "manual" | "auto" | False | "unknown"
    request_url: str = ""
    why: str = ""                    # short explanation when not accessible

    def asdict(self) -> dict:
        return asdict(self)


def current_token() -> str | None:
    """Read the user's HF token from env, HfFolder, or ~/.cache/huggingface."""
    if t := os.environ.get("HF_TOKEN"):
        return t.strip()
    if t := os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return t.strip()
    try:
        from huggingface_hub import HfFolder  # type: ignore
        t = HfFolder.get_token()
        return t.strip() if t else None
    except Exception:
        # Fallback: read the file directly so we don't hard-depend on the lib here.
        for p in (
            os.path.expanduser("~/.cache/huggingface/token"),
            os.path.expanduser("~/.huggingface/token"),
        ):
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        return f.read().strip()
                except OSError:
                    pass
    return None


def save_token(token: str) -> None:
    """Persist a validated token via the HF standard mechanism."""
    token = token.strip()
    if not token:
        raise ValueError("empty token")
    try:
        from huggingface_hub import HfFolder  # type: ignore
        HfFolder.save_token(token)
    except Exception:
        # Manual fallback to ~/.cache/huggingface/token
        path = os.path.expanduser("~/.cache/huggingface/token")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(token)
    # Also export to the current process env so daemon-issued
    # downloads pick it up immediately without a restart.
    os.environ["HF_TOKEN"] = token


def _http_get(url: str, token: str | None = None, timeout: float = 8.0) -> tuple[int, dict | None]:
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", "elmo-harness/0.1")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode("utf-8", errors="replace")
            try:
                return r.status, json.loads(body)
            except json.JSONDecodeError:
                return r.status, None
    except urllib.error.HTTPError as e:
        return e.code, None
    except (urllib.error.URLError, TimeoutError):
        return 0, None


def whoami(token: str | None = None) -> WhoAmI:
    """Validate a token. Returns username on success."""
    token = (token or current_token() or "").strip()
    if not token:
        return WhoAmI(configured=False)
    code, body = _http_get(f"{HF_API}/whoami-v2", token=token)
    if code == 200 and isinstance(body, dict):
        return WhoAmI(configured=True, username=body.get("name", ""))
    if code in (401, 403):
        return WhoAmI(configured=False, error="invalid or revoked token")
    return WhoAmI(configured=False, error=f"could not reach hugging face (http {code})")


def dataset_access(repo_id: str, token: str | None = None) -> DatasetAccess:
    """Check whether the current user can read `repo_id`. Heuristics:

    - GET /api/datasets/{repo_id} (no auth) → tells us if it's gated
    - GET /api/datasets/{repo_id} (with auth) → 200 if user can access,
      401/403 if user needs to request access or sign the terms
    """
    token = (token or current_token() or "").strip()
    request_url = f"https://huggingface.co/datasets/{repo_id}"

    pub_code, pub_body = _http_get(f"{HF_API}/datasets/{repo_id}")
    gated: str | bool = False
    if isinstance(pub_body, dict):
        g = pub_body.get("gated")
        if isinstance(g, bool):
            gated = g
        elif isinstance(g, str):
            gated = g

    auth_code, _ = _http_get(f"{HF_API}/datasets/{repo_id}", token=token)
    if auth_code == 200:
        return DatasetAccess(
            repo_id=repo_id, accessible=True, gated=gated, request_url=request_url,
        )
    if not token:
        return DatasetAccess(
            repo_id=repo_id, accessible=False, gated=gated or "unknown",
            request_url=request_url,
            why="no HF token configured. paste one to verify access.",
        )
    if auth_code in (401, 403):
        return DatasetAccess(
            repo_id=repo_id, accessible=False, gated=gated or "manual",
            request_url=request_url,
            why="access not yet granted — visit the dataset page and click 'Agree and access repository'.",
        )
    if auth_code == 404:
        return DatasetAccess(
            repo_id=repo_id, accessible=False, gated=False, request_url=request_url,
            why="dataset does not exist (typo in the spec?).",
        )
    return DatasetAccess(
        repo_id=repo_id, accessible=False, gated=gated or "unknown",
        request_url=request_url,
        why=f"could not check access (http {auth_code}).",
    )
