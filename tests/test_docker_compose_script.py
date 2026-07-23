from __future__ import annotations

import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "docker_compose.sh"


def _write_executable(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env bash\nset -eu\n{body}")
    path.chmod(0o755)


def _run_up(
    tmp_path: Path,
    tailscale_body: str | None,
) -> subprocess.CompletedProcess[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "bash").symlink_to("/usr/bin/bash")
    (bin_dir / "dirname").symlink_to("/usr/bin/dirname")
    if tailscale_body is not None:
        _write_executable(bin_dir / "tailscale", tailscale_body)
    _write_executable(bin_dir / "docker", 'printf "docker %s\\n" "$*"')

    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    return subprocess.run(
        [str(SCRIPT), "up", "--backend", "vggt", "--port", "9000", "--cpu", "--no-build"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_up_binds_compose_to_detected_tailscale_ipv4(tmp_path: Path) -> None:
    result = _run_up(
        tmp_path,
        """
if [ "$1" = "ip" ] && [ "$2" = "-4" ]; then
  printf '100.101.102.103\\n'
  exit 0
fi
exit 1
""",
    )

    assert result.returncode == 0
    assert "Binding GFM Serve to address: http://100.101.102.103:9000" in result.stdout
    assert "docker compose -f docker-compose.yml up" in result.stdout


def test_up_fails_when_tailscale_is_not_connected(tmp_path: Path) -> None:
    result = _run_up(tmp_path, "exit 1")

    assert result.returncode == 1
    assert "Tailscale is installed but is not connected." in result.stderr
    assert "docker compose" not in result.stdout


def test_up_fails_when_tailscale_is_not_installed(tmp_path: Path) -> None:
    result = _run_up(tmp_path, None)

    assert result.returncode == 1
    assert "Tailscale is required but the 'tailscale' command was not found." in result.stderr
    assert "docker compose" not in result.stdout


def test_up_rejects_invalid_tailscale_ipv4(tmp_path: Path) -> None:
    result = _run_up(tmp_path, "printf 'not-an-ip\\n'")

    assert result.returncode == 1
    assert "Could not detect a Tailscale IPv4 address." in result.stderr
    assert "docker compose" not in result.stdout
