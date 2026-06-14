"""Hardware probe — RAM, chip, GPU, available disk.

Stdlib only. Returns a structured snapshot the catalog uses to rank models.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass


@dataclass
class SystemProbe:
    os: str                       # "darwin" | "linux" | "windows"
    arch: str                     # "arm64" | "x86_64"
    chip: str                     # "Apple M5 Pro", "Intel i9-13900K", ...
    chip_class: str               # "apple-silicon" | "intel" | "amd" | "nvidia" | "other"
    chip_tier: str                # "base" | "pro" | "max" | "ultra" | "none"
    ram_gb: float
    free_disk_gb: float
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    suggested_backend: str = "none"  # "mlx" | "unsloth" | "none"

    def asdict(self) -> dict:
        return asdict(self)


def _sysctl(key: str) -> str:
    try:
        out = subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, timeout=2
        )
        return out.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _detect_apple_silicon_tier(chip: str) -> str:
    """Parse 'Apple M5 Pro' → 'pro'. Fallbacks to 'base' for plain Mx."""
    chip_low = chip.lower()
    for tier in ("ultra", "max", "pro"):
        if tier in chip_low:
            return tier
    if re.search(r"apple\s+m\d", chip_low):
        return "base"
    return "none"


def _detect_nvidia() -> tuple[str, float]:
    if not shutil.which("nvidia-smi"):
        return "", 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
    except subprocess.TimeoutExpired:
        return "", 0.0
    line = out.stdout.strip().splitlines()[0] if out.stdout.strip() else ""
    if not line:
        return "", 0.0
    parts = [p.strip() for p in line.split(",", 1)]
    if len(parts) != 2:
        return "", 0.0
    try:
        vram_mb = float(parts[1])
    except ValueError:
        vram_mb = 0.0
    return parts[0], vram_mb / 1024.0


def _free_disk_gb(path: str) -> float:
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except OSError:
        return 0.0


def probe() -> SystemProbe:
    os_name = platform.system().lower()
    arch = platform.machine().lower()

    chip = ""
    chip_class = "other"
    chip_tier = "none"
    ram_gb = 0.0
    gpu_name = ""
    gpu_vram = 0.0

    if os_name == "darwin":
        chip = _sysctl("machdep.cpu.brand_string") or "Unknown CPU"
        mem_str = _sysctl("hw.memsize")
        ram_gb = (int(mem_str) / (1024**3)) if mem_str.isdigit() else 0.0
        if "apple" in chip.lower() and arch == "arm64":
            chip_class = "apple-silicon"
            chip_tier = _detect_apple_silicon_tier(chip)
        else:
            chip_class = "intel" if "intel" in chip.lower() else "other"
    elif os_name == "linux":
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        ram_gb = kb / (1024**2)
                        break
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        chip = line.split(":", 1)[1].strip()
                        break
        except OSError:
            pass
        chip_class = "intel" if "intel" in chip.lower() else "amd" if "amd" in chip.lower() else "other"

    gpu_name, gpu_vram = _detect_nvidia()
    if gpu_name:
        chip_class = "nvidia"

    backend = "none"
    if chip_class == "apple-silicon":
        backend = "mlx"
    elif gpu_name:
        backend = "unsloth"

    return SystemProbe(
        os=os_name,
        arch=arch,
        chip=chip,
        chip_class=chip_class,
        chip_tier=chip_tier,
        ram_gb=round(ram_gb, 1),
        free_disk_gb=round(_free_disk_gb(os.path.expanduser("~")), 1),
        gpu_name=gpu_name,
        gpu_vram_gb=round(gpu_vram, 1),
        suggested_backend=backend,
    )
