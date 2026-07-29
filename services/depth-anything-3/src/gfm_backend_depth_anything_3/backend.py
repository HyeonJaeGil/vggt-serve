from __future__ import annotations

import logging
import threading
from time import perf_counter

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from gfm_serve.backends import BackendRunRequest, BackendRunResult, ReconstructionBackend
from gfm_serve.config import Settings
from gfm_serve.contracts import BackendDescriptor, CameraResult, ImageSize, SceneInput, ViewResult
from gfm_serve.errors import ServiceBusyApiError, ServiceUnavailableApiError
from gfm_serve.storage import (
    ArtifactDescriptor,
    sample_point_cloud,
    write_depth_artifact,
    write_point_cloud_ply,
)

from .config import DA3BackendSettings


LOGGER = logging.getLogger(__name__)


class DA3BackendOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    align_to_input_ext_scale: bool = True
    infer_gs: bool = False
    export_point_cloud: bool = True
    use_ray_pose: bool = False
    ref_view_strategy: str = Field(
        default="saddle_balanced",
        pattern="^(first|middle|saddle_balanced|saddle_sim_range)$",
    )
    process_res: int = Field(default=504, ge=224, le=2016)
    process_res_method: str = Field(
        default="upper_bound_resize",
        pattern="^(upper_bound_resize|lower_bound_resize)$",
    )


def _homogeneous_extrinsic(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape == (4, 4):
        return matrix
    if matrix.shape == (3, 4):
        result = np.eye(4, dtype=np.float32)
        result[:3, :] = matrix
        return result
    raise ValueError(f"DA3 returned an invalid extrinsic shape: {matrix.shape}.")


def _resize_float(array: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(np.asarray(array, dtype=np.float32), mode="F").resize(
            (width, height), resample=Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    )


def _points_from_depth(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    world_to_camera: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    y, x = np.indices((height, width), dtype=np.float32)
    z = depth.reshape(-1)
    pixels = np.stack((x.reshape(-1), y.reshape(-1), np.ones(height * width, dtype=np.float32)))
    camera_points = np.linalg.inv(intrinsics) @ pixels
    camera_points *= z[None, :]
    camera_h = np.concatenate((camera_points, np.ones((1, camera_points.shape[1]), dtype=np.float32)), axis=0)
    return (np.linalg.inv(world_to_camera) @ camera_h)[:3].T


class DA3Backend(ReconstructionBackend):
    backend_id = "depth-anything-3"
    display_name = "Depth Anything 3"
    capabilities = ("depth", "depth_confidence", "camera_poses", "point_cloud")
    options_model = DA3BackendOptions

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.backend_settings = DA3BackendSettings()
        self._model = None
        self._device: str | None = None
        self._last_error: str | None = None
        self._load_lock = threading.Lock()
        self._run_lock = threading.Lock()

    @property
    def descriptor(self) -> BackendDescriptor:
        inputs = {"images": {"required": True}}
        if self.backend_settings.supports_pose_input:
            inputs.update(
                {
                    "camera.intrinsics": {"required": False},
                    "camera.world_to_camera": {"required": False},
                }
            )
        outputs = list(self.capabilities)
        if self.backend_settings.supports_gaussians:
            outputs.append("depth-anything-3/gaussian-splats")
        return BackendDescriptor(
            backend=self.backend_id,
            model_id=self.backend_settings.model_id,
            model_revision=self.backend_settings.model_revision,
            inputs=inputs,
            outputs=outputs,
            options_schema=self.options_model.model_json_schema(),
        )

    @property
    def device_description(self) -> str | None:
        return self._device

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def is_ready(self) -> bool:
        return self._model is not None

    def validate_request(self, scene: SceneInput, options: dict[str, object] | None) -> BaseModel:
        validated = self.validate_options(options)
        cameras = [view.camera for view in scene.views]
        supplied = [camera is not None for camera in cameras]
        if any(supplied) and not all(supplied):
            raise ValueError("Pose-conditioned DA3 requests require camera data for every view.")
        if all(supplied):
            if not self.backend_settings.supports_pose_input:
                raise ValueError(f"Model '{self.backend_settings.model_id}' does not support pose input.")
            if any(
                camera is None or camera.intrinsics is None or camera.world_to_camera is None
                for camera in cameras
            ):
                raise ValueError("Each DA3 camera must include both intrinsics and world_to_camera.")
        if validated.infer_gs and not self.backend_settings.supports_gaussians:
            raise ValueError(f"Model '{self.backend_settings.model_id}' does not support infer_gs.")
        return validated

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from depth_anything_3.api import DepthAnything3

                configured = self.backend_settings.device
                self._device = (
                    configured
                    if configured != "auto"
                    else ("cuda" if torch.cuda.is_available() else "cpu")
                )
                self._model = DepthAnything3.from_pretrained(
                    self.backend_settings.model_id,
                    revision=self.backend_settings.model_revision,
                ).to(self._device)
                self._model.eval()
                self._last_error = None
            except Exception as exc:  # pragma: no cover - environment/model protection
                self._last_error = str(exc)
                raise

    def run(self, request: BackendRunRequest) -> BackendRunResult:
        if self._model is None:
            try:
                self.load()
            except Exception as exc:  # pragma: no cover - environment/model protection
                raise ServiceUnavailableApiError(str(exc)) from exc
        if self._model is None:
            raise ServiceUnavailableApiError(self._last_error or "Depth Anything 3 is not ready.")
        if not self._run_lock.acquire(blocking=False):
            raise ServiceBusyApiError()
        try:
            return self._run_locked(request)
        finally:
            self._run_lock.release()

    def _run_locked(self, request: BackendRunRequest) -> BackendRunResult:
        options = DA3BackendOptions.model_validate(request.backend_options)
        cameras = [view.camera for view in request.scene.views]
        pose_conditioned = all(camera is not None for camera in cameras)
        extrinsics = (
            np.asarray([camera.world_to_camera for camera in cameras], dtype=np.float32)
            if pose_conditioned
            else None
        )
        intrinsics = (
            np.asarray([camera.intrinsics for camera in cameras], dtype=np.float32)
            if pose_conditioned
            else None
        )
        start = perf_counter()
        prediction = self._model.inference(
            image=[str(view.image.path) for view in request.views],
            extrinsics=extrinsics,
            intrinsics=intrinsics,
            align_to_input_ext_scale=options.align_to_input_ext_scale,
            infer_gs=options.infer_gs,
            use_ray_pose=options.use_ray_pose,
            ref_view_strategy=options.ref_view_strategy,
            process_res=options.process_res,
            process_res_method=options.process_res_method,
        )
        inference_ms = int((perf_counter() - start) * 1000)
        postprocess_start = perf_counter()

        depths = np.asarray(prediction.depth, dtype=np.float32)
        has_confidence = prediction.conf is not None
        confidences = (
            np.asarray(prediction.conf, dtype=np.float32)
            if prediction.conf is not None
            else [np.ones_like(depth, dtype=np.float32) for depth in depths]
        )
        predicted_extrinsics = prediction.extrinsics
        predicted_intrinsics = prediction.intrinsics
        view_results: list[ViewResult] = []
        original_depths: list[np.ndarray] = []
        original_confidences: list[np.ndarray] = []
        all_points: list[np.ndarray] = []
        all_colors: list[np.ndarray] = []

        for index, view in enumerate(request.views):
            image = view.image
            depth = _resize_float(depths[index], image.width, image.height)
            confidence = _resize_float(np.asarray(confidences[index]), image.width, image.height)
            original_depths.append(depth)
            original_confidences.append(confidence)

            camera_result = None
            if predicted_extrinsics is not None or predicted_intrinsics is not None:
                world_to_camera = (
                    _homogeneous_extrinsic(predicted_extrinsics[index])
                    if predicted_extrinsics is not None
                    else None
                )
                intrinsic = (
                    np.asarray(predicted_intrinsics[index], dtype=np.float32).copy()
                    if predicted_intrinsics is not None
                    else None
                )
                if intrinsic is not None:
                    source_height, source_width = depths[index].shape
                    intrinsic[0, :] *= image.width / source_width
                    intrinsic[1, :] *= image.height / source_height
                source = "predicted"
                if pose_conditioned:
                    source = "provided" if options.align_to_input_ext_scale else "aligned"
                camera_result = CameraResult(
                    convention="opencv",
                    world_to_camera=world_to_camera.tolist() if world_to_camera is not None else None,
                    intrinsics=intrinsic.tolist() if intrinsic is not None else None,
                    source=source,
                )
                if (
                    options.export_point_cloud
                    and world_to_camera is not None
                    and intrinsic is not None
                ):
                    points = _points_from_depth(depth, intrinsic, world_to_camera)
                    colors = np.asarray(
                        Image.open(image.path).convert("RGB"), dtype=np.uint8
                    ).reshape(-1, 3)
                    flat_depth = depth.reshape(-1)
                    valid = (
                        np.isfinite(points).all(axis=1)
                        & np.isfinite(flat_depth)
                        & (flat_depth > 0)
                    )
                    all_points.append(points[valid])
                    all_colors.append(colors[valid])

            view_results.append(
                ViewResult(
                    view_id=view.view_id,
                    filename=image.original_filename,
                    original_size=ImageSize(width=image.width, height=image.height),
                    camera=camera_result,
                )
            )

        depth_path = request.run_dir / "depth.npz"
        write_depth_artifact(
            depth_path,
            [view.image.original_filename for view in request.views],
            [(view.image.width, view.image.height) for view in request.views],
            original_depths,
            original_confidences,
        )
        artifacts = [
            ArtifactDescriptor(
                name=depth_path.name,
                path=depth_path,
                kind="depth_archive",
                content_type="application/octet-stream",
                size_bytes=depth_path.stat().st_size,
                metadata={
                    "schema_version": "1.0",
                    "dtype": "float32",
                    "shape": ["views", "original_height", "original_width"],
                    "units": "metric" if bool(getattr(prediction, "is_metric", False)) else "model-relative",
                    "coordinate_convention": "opencv",
                    "backend": self.backend_id,
                    "model_id": self.backend_settings.model_id,
                    "model_revision": self.backend_settings.model_revision,
                },
            )
        ]
        outputs = ["depth"]
        if has_confidence:
            outputs.append("depth_confidence")
        if all(view.camera is not None for view in view_results):
            outputs.append("camera_poses")

        if all_points:
            points = np.concatenate(all_points)
            colors = np.concatenate(all_colors)
            points, colors = sample_point_cloud(
                points, colors, self.backend_settings.max_point_cloud_points
            )
            point_cloud_path = request.run_dir / "point_cloud.ply"
            write_point_cloud_ply(point_cloud_path, points, colors)
            artifacts.append(
                ArtifactDescriptor(
                    name=point_cloud_path.name,
                    path=point_cloud_path,
                    kind="point_cloud",
                    content_type="application/octet-stream",
                    size_bytes=point_cloud_path.stat().st_size,
                    metadata={
                        "schema_version": "1.0",
                        "dtype": "float32",
                        "shape": ["points", 3],
                        "units": "metric"
                        if bool(getattr(prediction, "is_metric", False))
                        else "model-relative",
                        "coordinate_convention": "opencv-world",
                        "backend": self.backend_id,
                        "model_id": self.backend_settings.model_id,
                        "model_revision": self.backend_settings.model_revision,
                    },
                )
            )
            outputs.append("point_cloud")

        if options.infer_gs and prediction.gaussians is not None:
            gaussian_path = request.run_dir / "gaussian_splats.npz"

            def as_numpy(value):
                return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)

            np.savez_compressed(
                gaussian_path,
                means=as_numpy(prediction.gaussians.means),
                scales=as_numpy(prediction.gaussians.scales),
                rotations=as_numpy(prediction.gaussians.rotations),
                harmonics=as_numpy(prediction.gaussians.harmonics),
                opacities=as_numpy(prediction.gaussians.opacities),
            )
            artifacts.append(
                ArtifactDescriptor(
                    name=gaussian_path.name,
                    path=gaussian_path,
                    kind="depth-anything-3/gaussian-splats",
                    content_type="application/octet-stream",
                    size_bytes=gaussian_path.stat().st_size,
                    metadata={
                        "schema_version": "1.0",
                        "coordinate_convention": "opencv-world",
                        "backend": self.backend_id,
                        "model_id": self.backend_settings.model_id,
                        "model_revision": self.backend_settings.model_revision,
                    },
                )
            )
            outputs.append("depth-anything-3/gaussian-splats")

        postprocess_ms = int((perf_counter() - postprocess_start) * 1000)
        LOGGER.info(
            "Depth Anything 3 inference completed",
            extra={"request_id": request.request_id, "image_count": len(request.views)},
        )
        return BackendRunResult(
            view_results=view_results,
            artifacts=artifacts,
            produced_outputs=outputs,
            timings_ms={
                "inference": inference_ms,
                "postprocess": postprocess_ms,
                "total": inference_ms + postprocess_ms,
            },
        )
