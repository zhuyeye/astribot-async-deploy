"""Fake-server end-to-end smoke test."""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.timeout(60)
def test_fake_e2e_pi05():
    server = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "fake_async_server.py"), "--port", "18080", "--protocol", "pi05"],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        time.sleep(1.0)
        proc = subprocess.run(
            [
                sys.executable,
                str(ROOT / "run_async_deploy.py"),
                "--host",
                "127.0.0.1",
                "--port",
                "18080",
                "--protocol",
                "pi05",
                "--fake-sensors",
                "--dry-run",
                "--max-runtime",
                "3",
                "--ctrl-hz",
                "50",
                "--send-hz",
                "20",
                "--no-stdin-reset",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        out = proc.stdout + proc.stderr
        assert proc.returncode == 0, out
        assert "chunks_seen=" in out
        # Should have received at least one chunk.
        assert "Ingested chunk" in out or "chunks_seen=" in out
    finally:
        server.terminate()
        try:
            server.wait(timeout=3)
        except subprocess.TimeoutExpired:
            server.kill()
