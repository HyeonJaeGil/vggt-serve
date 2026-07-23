from __future__ import annotations

import argparse
from pathlib import Path

from gfm_serve_client import VGGTClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Run VGGT through the Python SDK.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--base-url",
        required=True,
        help="Tailscale service URL printed by scripts/docker_compose.sh up.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vggt"))
    parser.add_argument("--depth-conf-threshold", type=float)
    args = parser.parse_args()

    with VGGTClient(args.base_url) as client:
        descriptor = client.model_descriptor()
        print(f"using {descriptor.model_id} ({descriptor.model_revision})")
        result = client.reconstruct(
            args.images,
            depth_conf_threshold=args.depth_conf_threshold,
            scene_id="vggt-example",
        )
        downloaded = client.download_artifacts(result, args.output_dir)

    print(f"request {result.request_id} completed in {result.timings_ms.total} ms")
    for path in downloaded:
        print(path)


if __name__ == "__main__":
    main()
