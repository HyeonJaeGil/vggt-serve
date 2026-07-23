from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gfm_serve_client import (
    CameraParameters,
    DepthAnything3Client,
    DepthAnything3Options,
)


def load_cameras(path: Path, image_count: int) -> list[CameraParameters]:
    """Load intrinsics[N,3,3] and either world_to_camera/camera_to_world[N,4,4]."""
    with np.load(path, allow_pickle=False) as data:
        intrinsics = data["intrinsics"]
        if "world_to_camera" in data:
            extrinsics = data["world_to_camera"]
            factory = CameraParameters.from_matrices
            extrinsic_name = "world_to_camera"
        elif "camera_to_world" in data:
            extrinsics = data["camera_to_world"]
            factory = CameraParameters.from_camera_to_world
            extrinsic_name = "camera_to_world"
        else:
            raise ValueError(
                "Camera NPZ must contain 'world_to_camera' or 'camera_to_world'."
            )

    if intrinsics.shape != (image_count, 3, 3):
        raise ValueError(f"intrinsics must have shape ({image_count}, 3, 3).")
    if extrinsics.shape != (image_count, 4, 4):
        raise ValueError(f"{extrinsic_name} must have shape ({image_count}, 4, 4).")
    return [
        factory(intrinsics=intrinsics[index], **{extrinsic_name: extrinsics[index]})
        for index in range(image_count)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DA3 through the Python SDK.")
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument(
        "--base-url",
        required=True,
        help="Tailscale service URL printed by scripts/docker_compose.sh up.",
    )
    parser.add_argument("--cameras", type=Path, help="Optional camera calibration NPZ.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/da3"))
    parser.add_argument("--process-res", type=int, default=504)
    args = parser.parse_args()

    cameras = load_cameras(args.cameras, len(args.images)) if args.cameras else None
    options = DepthAnything3Options(
        process_res=args.process_res,
        align_to_input_ext_scale=True,
    )

    with DepthAnything3Client(args.base_url) as client:
        descriptor = client.model_descriptor()
        print(f"using {descriptor.model_id} ({descriptor.model_revision})")
        result = client.reconstruct(
            args.images,
            cameras=cameras,
            options=options,
            scene_id="da3-example",
        )
        downloaded = client.download_artifacts(result, args.output_dir)

    print(f"request {result.request_id} completed in {result.timings_ms.total} ms")
    for path in downloaded:
        print(path)


if __name__ == "__main__":
    main()
