"""Background model download from Hugging Face with progress reporting.

Uses huggingface_hub's snapshot_download under the hood. The progress
heartbeat is written to a sidecar JSON file that the daemon reads to
broadcast over its WebSocket. Cancellation is cooperative via the same file.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

from elmo.hub import hub_root, register_base


@dataclass
class DownloadState:
    id: str
    hf_id: str
    display_name: str
    status: str                  # queued | downloading | done | error | cancelled
    bytes_downloaded: int = 0
    bytes_total: int = 0
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    error: str = ""
    path: str = ""


_lock = threading.Lock()
_active: dict[str, DownloadState] = {}


def _state_path(download_id: str) -> Path:
    return hub_root() / "downloads" / f"{download_id}.json"


def _persist(state: DownloadState) -> None:
    p = _state_path(state.id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(state)))


def get_state(download_id: str) -> DownloadState | None:
    p = _state_path(download_id)
    if not p.exists():
        return None
    return DownloadState(**json.loads(p.read_text()))


def list_active() -> list[DownloadState]:
    with _lock:
        return list(_active.values())


def _do_download(state: DownloadState, hf_id: str) -> None:
    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except ImportError as e:
        state.status = "error"
        state.error = f"huggingface_hub not installed: {e!r}"
        state.finished_at = time.time()
        _persist(state)
        return

    try:
        state.status = "downloading"
        _persist(state)
        # We don't have a great granular hook here; snapshot_download is mostly
        # opaque to byte counts unless we tee. We poll the cache dir to estimate.
        target_dir = hub_root() / "base" / hf_id.replace("/", "__")
        target_dir.mkdir(parents=True, exist_ok=True)

        # Spawn a poller that updates bytes_downloaded based on disk usage.
        stop_flag = threading.Event()

        def _poller():
            while not stop_flag.is_set():
                total = 0
                for p in target_dir.rglob("*"):
                    if p.is_file():
                        try:
                            total += p.stat().st_size
                        except OSError:
                            pass
                state.bytes_downloaded = total
                _persist(state)
                time.sleep(0.5)

        poller = threading.Thread(target=_poller, daemon=True)
        poller.start()

        local_path = snapshot_download(
            repo_id=hf_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
        )
        stop_flag.set()
        poller.join(timeout=2)

        # Final byte count
        total = 0
        for p in Path(local_path).rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        state.bytes_downloaded = total
        state.bytes_total = total
        state.status = "done"
        state.finished_at = time.time()
        state.path = str(local_path)
        _persist(state)

        register_base(hf_id=hf_id, display_name=state.display_name, path=local_path)
    except Exception as e:  # noqa: BLE001
        state.status = "error"
        state.error = repr(e)
        state.finished_at = time.time()
        _persist(state)
    finally:
        with _lock:
            _active.pop(state.id, None)


def start_download(hf_id: str, display_name: str, bytes_estimate: int = 0) -> DownloadState:
    state = DownloadState(
        id="dl_" + uuid.uuid4().hex[:10],
        hf_id=hf_id,
        display_name=display_name,
        status="queued",
        bytes_total=bytes_estimate,
    )
    _persist(state)
    with _lock:
        _active[state.id] = state
    t = threading.Thread(target=_do_download, args=(state, hf_id), daemon=True)
    t.start()
    return state
