# VGGT service

The VGGT service uses the pinned
[facebookresearch/VGGT](https://github.com/facebookresearch/vggt) upstream
commit recorded by `services/vggt/upstream`.

VGGT accepts image-only scenes. Supplied camera metadata is rejected. It
predicts OpenCV world-to-camera poses, pixel-space intrinsics, relative depth
and confidence, and a colored point cloud. Scale is model-relative.

The only request option is non-negative `depth_conf_threshold`. Configuration:

- `GFM_SERVE_VGGT_MODEL_ID` (default `facebook/VGGT-1B`)
- `GFM_SERVE_VGGT_MODEL_REVISION`
- `GFM_SERVE_VGGT_CHECKPOINT_PATH`
- `GFM_SERVE_VGGT_SQUARE_IMAGE_SIZE`
- `GFM_SERVE_VGGT_DEFAULT_DEPTH_CONF_THRESHOLD`
- `GFM_SERVE_VGGT_MAX_POINT_CLOUD_POINTS`

Build and run:

```bash
git submodule update --init --recursive
scripts/docker_compose.sh build --backend vggt
scripts/docker_compose.sh up --backend vggt
python scripts/client_example.py \
  services/vggt/upstream/examples/kitchen/images/00.png \
  --base-url http://100.101.102.103:8000 \
  --options-json '{"depth_conf_threshold": 1.0}'
```

Replace the example URL with the Tailscale URL printed by the startup command.
Startup fails if Tailscale is not installed and connected.

For application code, install `packages/gfm-serve-client` and use
`VGGTClient`. It exposes image inputs, typed VGGT options, result models, and
artifact downloads. See the
[Python SDK guide](../../packages/gfm-serve-client/README.md) and
[VGGT example](../../examples/vggt_client.py).

CUDA is strongly recommended. The image smoke check imports the pinned upstream
stack; numerical GPU integration remains opt-in.
