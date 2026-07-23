# Depth Anything 3 service

This service packages the pinned
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3)
upstream at commit `3fe327a6abe2e5db95b54444ea95463dbfef5610`.
The default checkpoint is `depth-anything/DA3NESTED-GIANT-LARGE`.

## Inputs and options

Image-only requests are supported by every configured variant. Pose-conditioned
requests must provide both OpenCV pixel-space `intrinsics` and homogeneous
world-to-camera `world_to_camera` matrices for every view. Mono and metric-only
variants reject camera input.

Supported options are `align_to_input_ext_scale`, `infer_gs`, `use_ray_pose`,
`ref_view_strategy`, `process_res`, and `process_res_method`. Unknown options
are rejected. `infer_gs` is advertised and accepted only for compatible giant
or nested giant variants.

The adapter returns original-resolution depth/confidence artifacts. It returns
OpenCV cameras and a point cloud when the selected variant produces both
extrinsics and intrinsics. Camera sources are marked `predicted`, `provided`,
or `aligned`.

The production image pins PyTorch `2.4.1`/Torchvision `0.19.1` with CUDA 12.1
and xFormers `0.0.28`. It installs only DA3 inference dependencies; upstream
CLI, COLMAP, Open3D UI, and development-only packages are intentionally omitted.

## Configuration

- `GFM_SERVE_DEPTH_ANYTHING_3_MODEL_ID`
- `GFM_SERVE_DEPTH_ANYTHING_3_MODEL_REVISION`
- `GFM_SERVE_DEPTH_ANYTHING_3_DEVICE` (`auto`, `cpu`, or `cuda`)
- `GFM_SERVE_DEPTH_ANYTHING_3_MAX_POINT_CLOUD_POINTS`

## Build and run

```bash
git submodule update --init --recursive
scripts/docker_compose.sh build --backend depth-anything-3
scripts/docker_compose.sh up --backend depth-anything-3
```

The startup command prints the service URL using the server's Tailscale IPv4
address and fails if Tailscale is not installed and connected.

Use the common manifest request documented in `docs/api.md`; select this image
with `GFM_SERVE_BACKEND=depth-anything-3`.

For application code, use `DepthAnything3Client` from
`packages/gfm-serve-client`. `CameraParameters.from_matrices()` accepts NumPy
world-to-camera matrices, while `CameraParameters.from_camera_to_world()`
performs pose inversion. Camera counts, shapes, finite values, and homogeneous
rows are checked before upload. See the
[Python SDK guide](../../packages/gfm-serve-client/README.md) and
[DA3 example](../../examples/depth_anything_3_client.py).

## Opt-in GPU smoke

The image build performs a dependency/import smoke without downloading weights.
On a CUDA host, run the separate end-to-end model smoke with a small fixture:

```bash
docker run --rm --gpus all \
  -v "$PWD/test-images:/fixtures:ro" \
  gfm-serve:depth-anything-3 \
  python /app/services/depth-anything-3/scripts/gpu_smoke.py \
  /fixtures/one.png /fixtures/two.png
```

It loads the configured checkpoint and verifies output view counts, shapes,
finiteness, and positive depth rather than unstable numeric equality.
