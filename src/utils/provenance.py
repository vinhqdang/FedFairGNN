"""Artifact provenance helper for FedFairGNN experiments.

Captures git commit hash, working directory dirty state, compute device,
timestamp (UTC ISO-8601), and runtime software environment metadata.
"""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, Tuple

try:
    import torch
    TORCH_VERSION = torch.__version__
except ImportError:
    TORCH_VERSION = "unknown"


def get_git_info() -> Tuple[str, bool]:
    """Return (git_commit, git_dirty). Reads environment variables first,
    falling back to running git in the current directory."""
    env_commit = os.environ.get("FEDFAIR_GIT_COMMIT") or os.environ.get("GIT_COMMIT")
    env_dirty = os.environ.get("FEDFAIR_GIT_DIRTY")
    if env_commit:
        dirty = (env_dirty == "1" or env_dirty == "true" or env_dirty == "True")
        return env_commit.strip(), dirty
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        # Exclude output results directory written during runs; manuscript/ must trigger dirty (ADR-14)
        status = subprocess.check_output(["git", "status", "--porcelain", "--", ".", ":!results"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = bool(status)
        return commit, dirty
    except Exception:
        return "unknown", False


def build_manifest(**extra: Any) -> Dict[str, Any]:
    """Build standardized provenance manifest dictionary for experimental artifacts."""
    commit, dirty = get_git_info()
    manifest = {
        "git_commit": commit,
        "git_dirty": dirty,
        "device": os.environ.get("FEDFAIR_DEVICE", "cpu"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "torch_version": TORCH_VERSION,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    manifest.update(extra)
    return manifest
