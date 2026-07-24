# VGGT service

VGGT reconstructs an image sequence without requiring camera calibration. It
returns model-relative depth, confidence, predicted OpenCV cameras, and a
colored point cloud.

## Quickstart

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
scripts/docker_compose.sh build --backend vggt
scripts/docker_compose.sh up --backend vggt
```

The startup command prints the service URL. In another terminal, replace the
example value with that URL and run:

```bash
python -m pip install -e packages/gfm-serve-client
export GFM_SERVE_URL="http://100.x.y.z:9000"

python examples/vggt_client.py \
  frames/*.png \
  --output-dir outputs/vggt
```

Image arguments are processed from left to right. Use filenames that sort in
view order when passing a shell glob.

The command prints the ordered inputs, selected model, request duration,
downloaded artifacts, and each depth map's shape. Its output directory looks
like this:

```text
outputs/vggt/
|-- depth.npz
|-- point_cloud.ply
`-- result.json
```

| File | Contents |
| --- | --- |
| `depth.npz` | Original-resolution depth and confidence for every input image |
| `point_cloud.ply` | Colored 3D points in the predicted OpenCV world frame |
| `result.json` | Model identity, cameras, timings, warnings, and artifact metadata |

Run `python examples/vggt_client.py --help` for all CLI arguments.

## Read the result

Depth maps retain each image's original resolution. Because views may have
different sizes, `depth` and `depth_conf` contain one NumPy array per view:

```python
import numpy as np

with np.load("outputs/vggt/depth.npz", allow_pickle=True) as result:
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
MeshLab. Depth and point-cloud scale are model-relative, not metric.

## Use from Python

The CLI above is a regular Python SDK example. The minimal application code is:

```python
import os

from gfm_serve_client import VGGTClient

with VGGTClient(os.environ["GFM_SERVE_URL"]) as client:
    result = client.reconstruct(
        ["frames/000.png", "frames/001.png"],
        scene_id="office",
        view_ids=["left", "right"],
        depth_conf_threshold=1.0,
    )
    paths = client.download_artifacts(result, "outputs/vggt")

for view in result.view_results:
    print(view.view_id, view.camera.world_to_camera_array)
```

VGGT predicts cameras, so `VGGTClient` intentionally has no supplied-camera
argument. See the [Python SDK guide](../../packages/gfm-serve-client/README.md)
for artifact selection, error handling, and typed result fields.

## CLI options

```bash
python examples/vggt_client.py frames/*.png \
  --base-url http://100.x.y.z:9000 \
  --depth-conf-threshold 1.0 \
  --scene-id office \
  --output-dir outputs/vggt \
  --overwrite
```

| Argument | Default | Meaning |
| --- | --- | --- |
| `images` | required | Ordered PNG/JPEG input paths |
| `--base-url` | `$GFM_SERVE_URL` | Running service URL |
| `--depth-conf-threshold` | server default | Exclude lower-confidence points from the point cloud |
| `--scene-id` | `vggt-example` | Identifier recorded in the result |
| `--output-dir` | `outputs/vggt` | Artifact download directory |
| `--overwrite` | off | Replace artifacts already in the output directory |

For a raw multipart HTTP request without the Python SDK, use
[`scripts/client_example.py`](../../scripts/client_example.py) or see the
[HTTP API guide](../../docs/api.md).

## Service configuration

| Environment variable | Meaning |
| --- | --- |
| `GFM_SERVE_VGGT_MODEL_ID` | Hugging Face model ID; default `facebook/VGGT-1B` |
| `GFM_SERVE_VGGT_MODEL_REVISION` | Optional pinned model revision |
| `GFM_SERVE_VGGT_CHECKPOINT_PATH` | Optional local checkpoint |
| `GFM_SERVE_VGGT_SQUARE_IMAGE_SIZE` | Model processing image size |
| `GFM_SERVE_VGGT_DEFAULT_DEPTH_CONF_THRESHOLD` | Default point-cloud confidence threshold |
| `GFM_SERVE_VGGT_MAX_POINT_CLOUD_POINTS` | Maximum exported point count |

The service uses the pinned
[facebookresearch/VGGT](https://github.com/facebookresearch/vggt) submodule
commit recorded at `services/vggt/upstream`. CUDA is strongly recommended.
Startup requires an installed and connected Tailscale client. The image build
checks imports; numerical GPU integration remains opt-in.
