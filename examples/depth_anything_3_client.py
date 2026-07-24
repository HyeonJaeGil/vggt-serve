from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from gfm_serve_client import (
    CameraParameters,
    DepthAnything3Client,
    DepthAnything3Options,
)


def load_cameras(path: Path, image_count: int) -> list[CameraParameters]:
    """Load cameras ordered exactly like the CLI image arguments."""
    with np.load(path, allow_pickle=False) as data:
        if "intrinsics" not in data:
            raise ValueError("Camera NPZ must contain 'intrinsics'.")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate depth from ordered images with Depth Anything 3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  export GFM_SERVE_URL=http://100.x.y.z:9000

  # Images only
  python examples/depth_anything_3_client.py frames/*.png

  # Images plus known cameras in matching order
  python examples/depth_anything_3_client.py scene/images/*.png \\
    --cameras scene/cameras.npz --output-dir outputs/da3
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
    parser.add_argument(
        "--cameras",
        type=Path,
        help=(
            "Optional NPZ with intrinsics[N,3,3] and either "
            "world_to_camera[N,4,4] or camera_to_world[N,4,4]."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/da3"))
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--process-res-method",
        choices=("upper_bound_resize", "lower_bound_resize"),
        default="upper_bound_resize",
    )
    parser.add_argument(
        "--ref-view-strategy",
        choices=("first", "middle", "saddle_balanced", "saddle_sim_range"),
        default="saddle_balanced",
    )
    parser.add_argument("--infer-gs", action="store_true")
    parser.add_argument("--use-ray-pose", action="store_true")
    parser.add_argument(
        "--align-to-input-ext-scale",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--scene-id", default="da3-example")
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
        print("\nDepth maps:")
        for filename, depth, confidence in zip(
            archive["filenames"],
            archive["depth"],
            archive["depth_conf"],
            strict=True,
        ):
            print(
                f"  {str(filename):<24} depth={tuple(depth.shape)!s:<14} "
                f"confidence={tuple(confidence.shape)} dtype={depth.dtype}"
            )


def main() -> None:
    args = parse_args()
    image_paths = [path.expanduser().resolve() for path in args.images]
    cameras = load_cameras(args.cameras, len(image_paths)) if args.cameras else None

    print(f"Input: {len(image_paths)} ordered image(s)")
    for index, path in enumerate(image_paths):
        print(f"  [{index}] {path}")
    if args.cameras:
        print(f"Cameras: {args.cameras.resolve()} ({len(cameras or [])} views)")
    else:
        print("Cameras: none (the model will predict supported geometry)")

    options = DepthAnything3Options(
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        ref_view_strategy=args.ref_view_strategy,
        infer_gs=args.infer_gs,
        use_ray_pose=args.use_ray_pose,
        align_to_input_ext_scale=args.align_to_input_ext_scale,
    )

    with DepthAnything3Client(args.base_url) as client:
        descriptor = client.model_descriptor()
        revision = descriptor.model_revision or "default revision"
        print(f"\nModel: {descriptor.model_id} ({revision})")
        result = client.reconstruct(
            image_paths,
            cameras=cameras,
            options=options,
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
            print(f"  {artifact.name:<20} {artifact.kind:<34} {size:>10}")

    depth_path = downloaded_by_name.get("depth.npz")
    if depth_path is not None:
        print_depth_summary(depth_path)

    camera_views = [view for view in result.view_results if view.camera is not None]
    if camera_views:
        sources = sorted({view.camera.source for view in camera_views})
        print(f"\nCameras: {len(camera_views)} returned; sources={', '.join(sources)}")


if __name__ == "__main__":
    main()
