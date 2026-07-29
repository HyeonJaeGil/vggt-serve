from __future__ import annotations

from io import BytesIO
from pathlib import Path

import httpx
import numpy as np
import pytest

from gfm_serve_client import (
    BackendMismatchError,
    CameraParameters,
    DepthAnything3Client,
    DepthAnything3Options,
    GFMServeAPIError,
    InvalidRequestError,
    VGGTClient,
)


def _descriptor(backend: str) -> dict:
    return {
        "backend": backend,
        "model_id": f"test/{backend}",
        "model_revision": "test",
        "inputs": {"images": {"required": True}},
        "outputs": ["depth"],
        "options_schema": {},
    }


def _result(backend: str, *, artifacts: list[dict] | None = None) -> dict:
    return {
        "service_version": "0.2.0",
        "result_schema_version": "1.0",
        "backend": backend,
        "model": _descriptor(backend),
        "request_id": "request-1",
        "status": "succeeded",
        "timings_ms": {"total": 10},
        "view_results": [
            {
                "view_id": "left",
                "filename": "left.png",
                "original_size": {"width": 2, "height": 2},
                "camera": None,
            }
        ],
        "artifacts": artifacts or [],
        "produced_outputs": ["depth"],
    }


def _image(path: Path) -> Path:
    path.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    return path


def test_camera_parameters_accept_numpy_and_invert_camera_to_world() -> None:
    camera_to_world = np.eye(4, dtype=np.float32)
    camera_to_world[0, 3] = 2
    camera = CameraParameters.from_camera_to_world(
        intrinsics=np.diag([500, 500, 1]),
        camera_to_world=camera_to_world,
    )

    np.testing.assert_allclose(camera.world_to_camera_array[0, 3], -2)
    np.testing.assert_allclose(camera.intrinsics_array, np.diag([500, 500, 1]))


@pytest.mark.parametrize(
    "world_to_camera,message",
    [
        (np.eye(3), "shape"),
        (np.full((4, 4), np.nan), "finite"),
        (np.zeros((4, 4)), "homogeneous"),
    ],
)
def test_camera_parameters_reject_invalid_extrinsics(
    world_to_camera: np.ndarray, message: str
) -> None:
    with pytest.raises(InvalidRequestError, match=message):
        CameraParameters.from_matrices(
            intrinsics=np.eye(3),
            world_to_camera=world_to_camera,
        )


def test_da3_client_builds_pose_conditioned_manifest(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(200, json=_descriptor("depth-anything-3"))
        return httpx.Response(200, json=_result("depth-anything-3"))

    camera = CameraParameters.from_matrices(
        intrinsics=np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]]),
        world_to_camera=np.eye(4),
    )
    images = [_image(tmp_path / "left.png"), _image(tmp_path / "right.png")]
    with DepthAnything3Client(
        "http://service.test", transport=httpx.MockTransport(handler)
    ) as client:
        result = client.reconstruct(
            images,
            cameras=[camera, camera],
            view_ids=["left", "right"],
            options=DepthAnything3Options(
                process_res=756,
                use_ray_pose=True,
                export_point_cloud=False,
            ),
        )

    assert result.backend == "depth-anything-3"
    body = requests[1].content.decode("utf-8", errors="replace")
    assert requests[1].url.path == "/v1/reconstructions"
    assert '"view_id":"left"' in body
    assert '"upload_key":"image_001"' in body
    assert '"world_to_camera":[[1.0,0.0,0.0,0.0]' in body
    assert '"process_res":756' in body
    assert '"use_ray_pose":true' in body
    assert '"export_point_cloud":false' in body
    assert 'name="image_000"; filename="left.png"' in body


def test_vggt_client_uses_typed_threshold_option(tmp_path: Path) -> None:
    bodies: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_descriptor("vggt"))
        bodies.append(request.content.decode("utf-8", errors="replace"))
        return httpx.Response(200, json=_result("vggt"))

    with VGGTClient(
        "http://service.test", transport=httpx.MockTransport(handler)
    ) as client:
        client.reconstruct(
            [_image(tmp_path / "image.png")],
            depth_conf_threshold=1.5,
        )

    assert '"depth_conf_threshold":1.5' in bodies[0]
    assert '"camera"' not in bodies[0]


def test_backend_specific_client_rejects_wrong_service() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_descriptor("depth-anything-3"))

    with VGGTClient(
        "http://service.test", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(BackendMismatchError):
            client.model_descriptor()


def test_structured_service_error_is_raised(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json=_descriptor("depth-anything-3"))
        return httpx.Response(
            422,
            json={
                "request_id": "bad-request",
                "status": "failed",
                "error": {"code": "validation_error", "message": "bad camera"},
            },
        )

    with DepthAnything3Client(
        "http://service.test", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(GFMServeAPIError) as raised:
            client.reconstruct([_image(tmp_path / "image.png")])

    assert raised.value.status_code == 422
    assert raised.value.code == "validation_error"
    assert raised.value.request_id == "bad-request"


def test_artifact_download_and_depth_loading(tmp_path: Path) -> None:
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        depth=np.ones((1, 2, 2), dtype=np.float32),
        confidence=np.full((1, 2, 2), 0.5, dtype=np.float32),
    )
    archive_bytes = buffer.getvalue()
    artifact = {
        "name": "depths.npz",
        "kind": "depth_archive",
        "url": "http://service.test/v1/artifacts/request-1/depths.npz",
        "content_type": "application/x-npz",
        "size_bytes": len(archive_bytes),
        "metadata": {},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/models/current":
            return httpx.Response(200, json=_descriptor("vggt"))
        if request.method == "POST":
            return httpx.Response(200, json=_result("vggt", artifacts=[artifact]))
        return httpx.Response(200, content=archive_bytes)

    with VGGTClient(
        "http://service.test", transport=httpx.MockTransport(handler)
    ) as client:
        result = client.reconstruct([_image(tmp_path / "image.png")])
        arrays = client.load_depth_archive(result)
        destination = client.download_artifact(
            result.artifact(kind="depth_archive"),
            tmp_path / "download",
        )

    np.testing.assert_array_equal(arrays["depth"], np.ones((1, 2, 2)))
    assert destination == tmp_path / "download"
    assert destination.read_bytes() == archive_bytes


def test_client_validates_counts_before_transport(tmp_path: Path) -> None:
    called = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal called
        called = True
        return httpx.Response(500)

    image = _image(tmp_path / "image.png")
    with DepthAnything3Client(
        "http://service.test", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(InvalidRequestError, match="cameras"):
            client.reconstruct([image], cameras=[])

    assert called is False


def test_not_ready_is_returned_as_status_instead_of_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={
                "status": "not_ready",
                "ready": False,
                "backend": "vggt",
                "capabilities": [],
                "error": "weights are loading",
            },
        )

    with VGGTClient(
        "http://service.test", transport=httpx.MockTransport(handler)
    ) as client:
        status = client.ready()

    assert status.ready is False
    assert status.error == "weights are loading"


def test_backend_mismatch_is_detected_before_upload(tmp_path: Path) -> None:
    methods: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        methods.append(request.method)
        return httpx.Response(200, json=_descriptor("vggt"))

    with DepthAnything3Client(
        "http://service.test", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(BackendMismatchError):
            client.reconstruct([_image(tmp_path / "image.png")])

    assert methods == ["GET"]
