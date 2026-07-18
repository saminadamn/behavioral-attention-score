"""Best-effort git commit detection for dataset provenance (Module 7, Step 7)."""

from __future__ import annotations

import subprocess


def detect_git_commit(cwd: str | None = None) -> str | None:
    """Return the current git commit hash, or `None` if unavailable.

    Never raises — not being in a git repository, git not being installed,
    or any other failure all just mean the provenance field stays empty,
    which is why Step 7 marks this field optional.
    """

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
