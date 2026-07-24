from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from gfm_serve_client import VGGTClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a scene from ordered images with VGGT.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""example:
  export GFM_SERVE_URL=http://100.x.y.z:9000
  python examples/vggt_client.py frames/*.png --output-dir outputs/vggt
""",
    )
    parser.add_argument(
        "images",
        nargs="+",
        type=Path,
        help="Ordered PNG/JPEG images. Shell globs such as frames/*.png are supported.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("GFM_SERVE_URL"),
        help="Service URL (default: GFM_SERVE_URL).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vggt"))
    parser.add_argument("--depth-conf-threshold", type=float)
    parser.add_argument("--scene-id", default="vggt-example")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing artifact files in --output-dir.",
    )
    args = parser.parse_args()
    if not args.base_url:
        parser.error("--base-url is required when GFM_SERVE_URL is not set")
    return args


def format_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024
    raise AssertionError("unreachable")


def print_depth_summary(path: Path) -> None:
    with np.load(path, allow_pickle=True) as archive:
        filenames = archive["filenames"]
        depths = archive["depth"]
        confidences = archive["depth_conf"]

        print("\nDepth maps:")
        for filename, depth, confidence in zip(
            filenames, depths, confidences, strict=True
        ):
            print(
                f"  {str(filename):<24} depth={tuple(depth.shape)!s:<14} "
                f"confidence={tuple(confidence.shape)} dtype={depth.dtype}"
            )


def main() -> None:
    args = parse_args()
    image_paths = [path.expanduser().resolve() for path in args.images]

    print(f"Input: {len(image_paths)} ordered image(s)")
    for index, path in enumerate(image_paths):
        print(f"  [{index}] {path}")

    with VGGTClient(args.base_url) as client:
        descriptor = client.model_descriptor()
        revision = descriptor.model_revision or "default revision"
        print(f"\nModel: {descriptor.model_id} ({revision})")
        result = client.reconstruct(
            image_paths,
            depth_conf_threshold=args.depth_conf_threshold,
            scene_id=args.scene_id,
        )
        downloaded = client.download_artifacts(
            result,
            args.output_dir,
            overwrite=args.overwrite,
        )

    print(
        f"\nCompleted: request={result.request_id} "
        f"total={result.timings_ms.total} ms"
    )
    print(f"Output directory: {args.output_dir.resolve()}")
    downloaded_by_name = {path.name: path for path in downloaded}
    for artifact in result.artifacts:
        path = downloaded_by_name.get(artifact.name)
        if path is not None:
            size = format_bytes(artifact.size_bytes)
            print(f"  {artifact.name:<20} {artifact.kind:<16} {size:>10}")

    depth_path = downloaded_by_name.get("depth.npz")
    if depth_path is not None:
        print_depth_summary(depth_path)

    cameras = [view for view in result.view_results if view.camera is not None]
    if cameras:
        first = cameras[0]
        print(
            f"\nCameras: {len(cameras)} predicted; first view={first.view_id!r}, "
            f"source={first.camera.source}"
        )


if __name__ == "__main__":
    main()
