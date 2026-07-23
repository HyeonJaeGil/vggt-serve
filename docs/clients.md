# Choose a client

Use the typed Python SDK for Python applications. Use the HTTP API directly
for shell scripts and other languages. In every case, use the URL printed by
`scripts/docker_compose.sh up`; it contains the server's Tailscale IPv4 address
and is reachable from clients on the same tailnet.

| Use case | Start here |
| --- | --- |
| Python application | [Python SDK guide](../packages/gfm-serve-client/README.md) |
| VGGT Python example | [`examples/vggt_client.py`](../examples/vggt_client.py) |
| DA3 Python example | [`examples/depth_anything_3_client.py`](../examples/depth_anything_3_client.py) |
| `curl` or another language | [HTTP API and curl examples](api.md) |

The Python SDK provides `VGGTClient`, `DepthAnything3Client`, camera validation,
typed results, and artifact downloads. The HTTP API exposes the same features
as multipart requests.
