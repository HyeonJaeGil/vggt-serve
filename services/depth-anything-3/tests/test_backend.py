from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from gfm_backend_depth_anything_3 import DA3Backend, DA3BackendOptions
from gfm_backend_depth_anything_3.backend import _homogeneous_extrinsic
from gfm_backend_depth_anything_3.config import DA3BackendSettings
from gfm_serve.backends import BackendRunRequest, PreparedView
from gfm_serve.config import Settings
from gfm_serve.contracts import SceneInput
from gfm_serve.storage import PreparedImage


IDENTITY_4 = np.eye(4, dtype=np.float32).tolist()
IDENTITY_3 = np.eye(3, dtype=np.float32).tolist()


def _scene(camera_payloads: list[dict | None]) -> SceneInput:
    return SceneInput.model_validate(
        {
            "views": [
                {"view_id": f"view-{index}", "upload_key": f"image-{index}", "camera": camera}
                for index, camera in enumerate(camera_payloads)
            ]
        }
    )


def _backend(tmp_path: Path, monkeypatch, model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE") -> DA3Backend:
    monkeypatch.setenv("GFM_SERVE_DEPTH_ANYTHING_3_MODEL_ID", model_id)
    return DA3Backend(Settings(data_root=tmp_path / "runs"))


def test_images_only_request_is_valid(tmp_path: Path, monkeypatch) -> None:
    backend = _backend(tmp_path, monkeypatch)

    options = backend.validate_request(_scene([None, None]), {"ref_view_strategy": "middle"})

    assert options.ref_view_strategy == "middle"


@pytest.mark.parametrize(
    "cameras,message",
    [
        ([{"convention": "opencv", "world_to_camera": IDENTITY_4, "intrinsics": IDENTITY_3}, None], "every view"),
        (
            [
                {"convention": "opencv", "world_to_camera": IDENTITY_4},
                {"convention": "opencv", "world_to_camera": IDENTITY_4},
            ],
            "both intrinsics",
        ),
    ],
)
def test_partial_camera_input_is_rejected(
    tmp_path: Path, monkeypatch, cameras: list[dict | None], message: str
) -> None:
    backend = _backend(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match=message):
        backend.validate_request(_scene(cameras), {})


def test_variant_restricted_gaussian_option_is_rejected(tmp_path: Path, monkeypatch) -> None:
    backend = _backend(tmp_path, monkeypatch, model_id="depth-anything/DA3-LARGE")

    with pytest.raises(ValueError, match="does not support infer_gs"):
        backend.validate_request(_scene([None]), {"infer_gs": True})


def test_unknown_options_are_rejected() -> None:
    with pytest.raises(ValidationError):
        DA3BackendOptions.model_validate({"upstream_escape_hatch": True})


def test_extrinsic_3x4_is_normalized_to_homogeneous_matrix() -> None:
    normalized = _homogeneous_extrinsic(np.eye(4, dtype=np.float32)[:3])

    np.testing.assert_array_equal(normalized[3], [0, 0, 0, 1])


def test_pose_conditioned_stub_execution_forwards_ordered_arrays(tmp_path: Path, monkeypatch) -> None:
    backend = _backend(tmp_path, monkeypatch)
    calls: list[dict] = []

    class StubModel:
        def inference(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                depth=np.ones((2, 3, 4), dtype=np.float32),
                conf=np.full((2, 3, 4), 2.0, dtype=np.float32),
                extrinsics=np.stack([np.eye(4, dtype=np.float32)[:3]] * 2),
                intrinsics=np.stack([np.eye(3, dtype=np.float32)] * 2),
            )

    backend._model = StubModel()
    backend._device = "cpu"
    camera_a = {"convention": "opencv", "world_to_camera": IDENTITY_4, "intrinsics": IDENTITY_3}
    camera_b = {
        "convention": "opencv",
        "world_to_camera": (np.eye(4, dtype=np.float32) * 2).tolist(),
        "intrinsics": (np.eye(3, dtype=np.float32) * 3).tolist(),
    }
    scene = _scene([camera_a, camera_b])
    views = []
    for index, view in enumerate(scene.views):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (4, 3), color=(index * 10, 0, 0)).save(path)
        image = PreparedImage(
            original_filename=path.name,
            stored_filename=path.name,
            path=path,
            size_bytes=path.stat().st_size,
            width=4,
            height=3,
            content_type="image/png",
        )
        views.append(PreparedView(view_id=view.view_id, upload_key=view.upload_key, image=image))
    options = backend.validate_request(scene, {"align_to_input_ext_scale": False})

    result = backend.run(
        BackendRunRequest(
            request_id="test",
            run_dir=tmp_path,
            scene=scene,
            views=views,
            backend_options=options,
        )
    )

    assert calls[0]["image"] == [str(tmp_path / "0.png"), str(tmp_path / "1.png")]
    np.testing.assert_array_equal(calls[0]["extrinsics"][1], np.eye(4) * 2)
    np.testing.assert_array_equal(calls[0]["intrinsics"][1], np.eye(3) * 3)
    assert calls[0]["align_to_input_ext_scale"] is False
    assert [view.view_id for view in result.view_results] == ["view-0", "view-1"]
    assert all(view.camera.source == "aligned" for view in result.view_results)
    assert {artifact.kind for artifact in result.artifacts} == {"depth_archive", "point_cloud"}


def test_settings_report_variant_capabilities(monkeypatch) -> None:
    monkeypatch.setenv("GFM_SERVE_DEPTH_ANYTHING_3_MODEL_ID", "depth-anything/DA3MONO-LARGE")

    settings = DA3BackendSettings()

    assert settings.supports_pose_input is False
    assert settings.supports_gaussians is False


def test_point_cloud_export_can_be_disabled(tmp_path: Path, monkeypatch) -> None:
    backend = _backend(tmp_path, monkeypatch)

    class StubModel:
        def inference(self, **kwargs):
            return SimpleNamespace(
                depth=np.ones((1, 3, 4), dtype=np.float32),
                conf=np.ones((1, 3, 4), dtype=np.float32),
                extrinsics=np.stack([np.eye(4, dtype=np.float32)[:3]]),
                intrinsics=np.stack([np.eye(3, dtype=np.float32)]),
            )

    backend._model = StubModel()
    backend._device = "cpu"
    camera = {"convention": "opencv", "world_to_camera": IDENTITY_4, "intrinsics": IDENTITY_3}
    scene = _scene([camera])
    path = tmp_path / "frame.png"
    Image.new("RGB", (4, 3)).save(path)
    view = PreparedView(
        view_id=scene.views[0].view_id,
        upload_key=scene.views[0].upload_key,
        image=PreparedImage(
            original_filename=path.name,
            stored_filename=path.name,
            path=path,
            size_bytes=path.stat().st_size,
            width=4,
            height=3,
            content_type="image/png",
        ),
    )
    options = backend.validate_request(scene, {"export_point_cloud": False})

    result = backend.run(
        BackendRunRequest(
            request_id="compact",
            run_dir=tmp_path,
            scene=scene,
            views=[view],
            backend_options=options,
        )
    )

    assert {artifact.kind for artifact in result.artifacts} == {"depth_archive"}
    assert "camera_poses" in result.produced_outputs
    assert "point_cloud" not in result.produced_outputs
