# Depth Anything 3 service

Depth Anything 3 (DA3) estimates depth from one or more ordered images. With a
compatible checkpoint it can also predict cameras and a colored point cloud,
or align the reconstruction to cameras supplied by the caller.

## Quickstart: images only

Prepare ordered views of one scene:

```text
frames/
|-- 000.png
|-- 001.png
`-- 002.png
```

Start the service:

```bash
git submodule update --init --recursive
scripts/docker_compose.sh build --backend depth-anything-3
scripts/docker_compose.sh up --backend depth-anything-3
```

The startup command prints the service URL. In another terminal, replace the
example value with that URL and run:

```bash
python -m pip install -e packages/gfm-serve-client
export GFM_SERVE_URL="http://100.x.y.z:9000"

python examples/depth_anything_3_client.py \
  frames/*.png \
  --output-dir outputs/da3
```

Image arguments are processed from left to right. Use filenames that sort in
view order when passing a shell glob.

The command prints the ordered inputs, selected model, request duration,
downloaded artifacts, and each depth map's shape. Depending on the checkpoint
and options, the output directory contains:

```text
outputs/da3/
|-- depth.npz
|-- point_cloud.ply       # when the model returns cameras
|-- gaussian_splats.npz   # only with --infer-gs and a compatible model
`-- result.json
```

| File | Contents |
| --- | --- |
| `depth.npz` | Original-resolution depth and confidence for every input image |
| `point_cloud.ply` | Colored 3D points in an OpenCV world frame |
| `gaussian_splats.npz` | Gaussian means, scales, rotations, harmonics, and opacities |
| `result.json` | Model identity, cameras, timings, warnings, and artifact metadata |

Run `python examples/depth_anything_3_client.py --help` for all CLI arguments.

## Quickstart: known cameras

Pose-conditioned inference expects a camera for every image, in exactly the
same order:

```text
scene/
|-- images/
|   |-- 000.png
|   `-- 001.png
`-- cameras.npz
```

`cameras.npz` must use one of these layouts:

```text
intrinsics       (N, 3, 3)  +  world_to_camera (N, 4, 4)
intrinsics       (N, 3, 3)  +  camera_to_world (N, 4, 4)
```

Create a two-view example:

```python
import numpy as np

intrinsics = np.array(
    [
        [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
        [[800, 0, 640], [0, 800, 360], [0, 0, 1]],
    ],
    dtype=np.float32,
)
camera_to_world = np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0)
camera_to_world[1, 0, 3] = 0.1

np.savez(
    "scene/cameras.npz",
    intrinsics=intrinsics,
    camera_to_world=camera_to_world,
)
```

Then pass the images and camera archive together:

```bash
python examples/depth_anything_3_client.py \
  scene/images/*.png \
  --cameras scene/cameras.npz \
  --output-dir outputs/da3
```

Camera input rules:

- the camera array order must match the image argument order;
- `intrinsics` are pixel-space values for each original uploaded image;
- extrinsics are finite homogeneous `4×4` matrices;
- `world_to_camera` follows `X_camera = R @ X_world + t`;
- every view must have both intrinsics and an extrinsic;
- mono and metric-only DA3 variants reject supplied cameras.

The client converts `camera_to_world` to `world_to_camera` before upload.
VGGT does not accept supplied cameras.

## Read the result

Depth maps retain each image's original resolution. Because views may have
different sizes, `depth` and `depth_conf` contain one NumPy array per view:

```python
import numpy as np

with np.load("outputs/da3/depth.npz", allow_pickle=True) as result:
    print(result.files)
    # ['filenames', 'image_sizes', 'depth', 'depth_conf']

    for filename, depth, confidence in zip(
        result["filenames"],
        result["depth"],
        result["depth_conf"],
        strict=True,
    ):
        print(filename, depth.shape, confidence.shape, depth.dtype)
```

Open `point_cloud.ply` with a PLY-compatible viewer such as CloudCompare or
MeshLab. Check the `units` metadata in `result.json`: it is `metric` only when
the selected checkpoint produces metric depth; otherwise it is
`model-relative`.

## Use from Python

Image-only inference:

```python
import os

from gfm_serve_client import DepthAnything3Client, DepthAnything3Options

with DepthAnything3Client(os.environ["GFM_SERVE_URL"]) as client:
    result = client.reconstruct(
        ["frames/000.png", "frames/001.png"],
        options=DepthAnything3Options(
            process_res=756,
            ref_view_strategy="middle",
        ),
    )
    paths = client.download_artifacts(result, "outputs/da3")
```

For known cameras, construct one `CameraParameters` per image:

```python
import os

import numpy as np

from gfm_serve_client import CameraParameters, DepthAnything3Client

with np.load("scene/cameras.npz") as data:
    cameras = [
        CameraParameters.from_camera_to_world(
            intrinsics=intrinsics,
            camera_to_world=camera_to_world,
        )
        for intrinsics, camera_to_world in zip(
            data["intrinsics"],
            data["camera_to_world"],
            strict=True,
        )
    ]

with DepthAnything3Client(os.environ["GFM_SERVE_URL"]) as client:
    result = client.reconstruct(
        ["scene/images/000.png", "scene/images/001.png"],
        cameras=cameras,
    )
```

See the [Python SDK guide](../../packages/gfm-serve-client/README.md) for
artifact selection, errors, and typed result fields.

## CLI options

```bash
python examples/depth_anything_3_client.py frames/*.png \
  --base-url http://100.x.y.z:9000 \
  --process-res 504 \
  --ref-view-strategy saddle_balanced \
  --output-dir outputs/da3 \
  --overwrite
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `images` | required | Ordered PNG/JPEG input paths |
| `--base-url` | `$GFM_SERVE_URL` | Running service URL |
| `--cameras` | none | Camera NPZ described above |
| `--output-dir` | `outputs/da3` | Artifact download directory |
| `--process-res` | `504` | Model processing resolution |
| `--process-res-method` | `upper_bound_resize` | Processing resize rule |
| `--ref-view-strategy` | `saddle_balanced` | Reference-view selection strategy |
| `--infer-gs` | off | Request Gaussian splats from a compatible model |
| `--use-ray-pose` | off | Use ray-based pose conditioning |
| `--[no-]align-to-input-ext-scale` | on | Align predicted geometry scale to supplied extrinsics |
| `--scene-id` | `da3-example` | Identifier recorded in the result |
| `--overwrite` | off | Replace artifacts already in the output directory |

Unknown options are rejected. `infer_gs` is accepted only by compatible giant
or nested-giant variants. For a raw multipart HTTP request without the Python
SDK, use [`scripts/client_example.py`](../../scripts/client_example.py) or see
the [HTTP API guide](../../docs/api.md).

## Service configuration

| Environment variable | Meaning |
| --- | --- |
| `GFM_SERVE_DEPTH_ANYTHING_3_MODEL_ID` | Model ID; default `depth-anything/DA3NESTED-GIANT-LARGE` |
| `GFM_SERVE_DEPTH_ANYTHING_3_MODEL_REVISION` | Optional pinned model revision |
| `GFM_SERVE_DEPTH_ANYTHING_3_DEVICE` | `auto`, `cpu`, or `cuda` |
| `GFM_SERVE_DEPTH_ANYTHING_3_MAX_POINT_CLOUD_POINTS` | Maximum exported point count |

The service packages
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) at commit
`3fe327a6abe2e5db95b54444ea95463dbfef5610`. The production image pins PyTorch
2.4.1, Torchvision 0.19.1, CUDA 12.1, and xFormers 0.0.28. It omits the upstream
CLI, COLMAP, Open3D UI, and development-only packages. Startup requires an
installed and connected Tailscale client.

## Opt-in GPU smoke test

The image build checks dependencies and imports without downloading weights.
On a CUDA host, run an end-to-end inference smoke with a small fixture:

```bash
docker run --rm --gpus all \
  -v "$PWD/test-images:/fixtures:ro" \
  gfm-serve:depth-anything-3 \
  python /app/services/depth-anything-3/scripts/gpu_smoke.py \
  /fixtures/one.png /fixtures/two.png
```

The smoke test verifies view counts, shapes, finite values, and positive depth;
it does not depend on unstable numeric equality.
