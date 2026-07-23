from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test client for the reconstruction service.")
    parser.add_argument(
        "images",
        nargs="+",
        help="Image paths to upload as one reconstruction request.",
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Tailscale service URL printed by scripts/docker_compose.sh up.",
    )
    parser.add_argument(
        "--scene-id",
        default=None,
        help="Optional scene identifier to send with the request.",
    )
    parser.add_argument(
        "--client-request-id",
        default=None,
        help="Optional client request identifier to send with the request.",
    )
    parser.add_argument(
        "--options-json",
        default=None,
        help="Optional JSON object passed as manifest options.",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="If set, download all returned artifacts into this directory.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def guess_content_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    raise ValueError(f"Unsupported file extension for upload: {path}")


def build_files(
    image_paths: Iterable[Path],
) -> list[tuple[str, tuple[str, object, str]]]:
    files: list[tuple[str, tuple[str, object, str]]] = []
    for image_path in image_paths:
        files.append(
            (
                f"image_{len(files):03d}",
                (
                    image_path.name,
                    image_path.open("rb"),
                    guess_content_type(image_path),
                ),
            )
        )
    return files


def close_files(files: list[tuple[str, tuple[str, object, str]]]) -> None:
    for _, (_, handle, _) in files:
        handle.close()


def download_artifacts(client: httpx.Client, payload: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for artifact in payload.get("artifacts", []):
        url = artifact["url"]
        name = artifact["name"]
        destination = output_dir / name
        with client.stream("GET", url) as response:
            response.raise_for_status()
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes():
                    handle.write(chunk)
        print(f"downloaded {name} -> {destination}")


def main() -> int:
    args = parse_args()
    image_paths = [Path(path).expanduser().resolve() for path in args.images]
    for image_path in image_paths:
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

    files = build_files(image_paths)
    options = json.loads(args.options_json) if args.options_json else {}
    if not isinstance(options, dict):
        raise ValueError("--options-json must contain a JSON object.")
    data = {
        "manifest": json.dumps(
            {
                "scene_id": args.scene_id,
                "views": [
                    {"view_id": f"view-{index:03d}", "upload_key": key}
                    for index, (key, _) in enumerate(files)
                ],
                "options": options,
            }
        )
    }
    if args.client_request_id is not None:
        data["client_request_id"] = args.client_request_id

    url = f"{args.base_url.rstrip('/')}/v1/reconstructions"
    try:
        with httpx.Client(timeout=args.timeout) as client:
            response = client.post(url, data=data, files=files)
            payload = response.json()
            print(json.dumps(payload, indent=2))
            response.raise_for_status()

            if args.download_dir is not None:
                request_id = payload["request_id"]
                download_dir = args.download_dir / request_id
                download_artifacts(client, payload, download_dir)
    finally:
        close_files(files)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
