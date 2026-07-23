# GFM Serve API

The examples expect the Tailscale URL printed when the server starts:

```bash
export GFM_SERVE_URL="http://100.101.102.103:9000"
```

Replace the example IP with the server's `tailscale ip -4` result.

## Discovery and health

- `GET /healthz` reports HTTP process health.
- `GET /readyz` reports the selected backend's concise readiness state.
- `GET /v1/models/current` returns the exact model/checkpoint descriptor,
  accepted common inputs, produced outputs, and the backend options JSON Schema.

## Create a reconstruction

`POST /v1/reconstructions` accepts multipart form data with one JSON `manifest`
field and one file field per image.

```bash
curl -X POST "$GFM_SERVE_URL/v1/reconstructions" \
  -F 'manifest={
    "scene_id":"office",
    "views":[
      {"view_id":"left","upload_key":"image_000"},
      {"view_id":"right","upload_key":"image_001"}
    ],
    "options":{}
  }' \
  -F image_000=@frames/000.png \
  -F image_001=@frames/001.png
```

This image-only request works with both VGGT and DA3.

### DA3 with known cameras

Add a camera to every view and pass both matrices:

```bash
curl -X POST "$GFM_SERVE_URL/v1/reconstructions" \
  -F 'manifest={
    "scene_id":"office",
    "views":[
      {
        "view_id":"left",
        "upload_key":"image_000",
        "camera":{
          "convention":"opencv",
          "world_to_camera":[
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
          ],
          "intrinsics":[
            [800,0,640],
            [0,800,360],
            [0,0,1]
          ]
        }
      },
      {
        "view_id":"right",
        "upload_key":"image_001",
        "camera":{
          "convention":"opencv",
          "world_to_camera":[
            [1,0,0,-0.1],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
          ],
          "intrinsics":[
            [800,0,640],
            [0,800,360],
            [0,0,1]
          ]
        }
      }
    ],
    "options":{
      "align_to_input_ext_scale":true,
      "process_res":504
    }
  }' \
  -F image_000=@frames/000.png \
  -F image_001=@frames/001.png
```

File order is irrelevant; `upload_key` determines view order. View IDs and
upload keys must be unique. Extra, missing, duplicated, non-image, empty,
corrupt, and oversized file parts are rejected.

Camera rules:

- use OpenCV coordinates;
- provide finite homogeneous `4×4` world-to-camera extrinsics;
- provide pixel-space `3×3` intrinsics for the original uploaded image size;
- provide both matrices for every view;
- use a DA3 checkpoint that supports pose input. VGGT, DA3 mono, and DA3
  metric-only variants reject supplied cameras.

## Results and artifacts

Per-view results are keyed by stable `view_id`. Camera matrices are optional and
carry a `source` of `predicted`, `provided`, or `aligned`. Large arrays remain
in downloadable artifacts.

`result.json` records the validated request, selected model, outputs, artifacts,
timings, and warnings. `depth_archive` artifacts contain float32
original-resolution depth and confidence arrays. Point clouds use the
OpenCV-derived world frame.

Artifacts are retrieved from
`GET /v1/artifacts/{request_id}/{name}` and are confined to the request run
directory.

Python applications can avoid multipart handling by using the
[Python SDK](../packages/gfm-serve-client/README.md).
