# GFM Serve

GFM Serve exposes geometric foundation models through one stable reconstruction
API while keeping each model's options, dependencies, and GPU ownership
isolated. One service process loads exactly one backend.

| Backend | Service documentation | Inputs |
| --- | --- | --- |
| VGGT | [services/vggt/README.md](services/vggt/README.md) | images |
| Depth Anything 3 | [services/depth-anything-3/README.md](services/depth-anything-3/README.md) | images, optional cameras |

## Quick start

GFM Serve is exposed only on the host's Tailscale IPv4 address. Install
Tailscale, join the server and client machines to the same tailnet, and confirm
the server is connected:

```bash
tailscale status
tailscale ip -4
```

The startup wrapper checks both the CLI installation and the connection before
starting Docker:

```bash
git submodule update --init --recursive
scripts/docker_compose.sh up --backend vggt --port 9000
# or
scripts/docker_compose.sh up --backend depth-anything-3 --port 9000
```

On the server, capture the URL printed by the wrapper (for example,
`http://100.101.102.103:9000`) or construct it with:

```bash
export GFM_SERVE_URL="http://$(tailscale ip -4 | head -n 1):9000"
```

Inspect the active checkpoint and its variant-dependent capabilities:

```bash
curl "$GFM_SERVE_URL/readyz"
curl "$GFM_SERVE_URL/v1/models/current"
```

Install the Python SDK and run an example:

```bash
python -m pip install -e packages/gfm-serve-client
python examples/vggt_client.py \
  services/vggt/upstream/examples/kitchen/images/00.png \
  services/vggt/upstream/examples/kitchen/images/01.png \
  --base-url "$GFM_SERVE_URL" \
  --output-dir ./client_outputs
```

Applications should use the typed
[Python client](packages/gfm-serve-client/README.md). DA3 camera input and
artifact handling are covered there. For `curl`, other languages, and all
available examples, see [Choose a client](docs/clients.md).

## Development

```bash
python -m pip install -e 'packages/gfm-serve-core[test]'
python -m pip install -e 'packages/gfm-serve-client[test]'
python -m pip install --no-deps -e services/vggt
python -m pip install --no-deps -e services/depth-anything-3
pytest -q
```

For a local model environment, install only the upstream and service package
you intend to run. Start the app with:

```bash
GFM_SERVE_BACKEND=vggt uvicorn --factory gfm_serve.app:create_app \
  --host "$(tailscale ip -4 | head -n 1)" --port 8000
```

Architecture and extension rules are in
[docs/architecture.md](docs/architecture.md).
