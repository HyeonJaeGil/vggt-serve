from __future__ import annotations

import math
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import InvalidRequestError


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]
Matrix3x3 = tuple[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat],
]
Matrix4x4 = tuple[
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
    tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat],
]


def _matrix(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise InvalidRequestError(f"{name} must have shape {shape}; got {array.shape}.")
    if not np.isfinite(array).all():
        raise InvalidRequestError(f"{name} must contain only finite values.")
    return array


class CameraParameters(BaseModel):
    """OpenCV camera calibration for one original-resolution image."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    intrinsics: Matrix3x3
    world_to_camera: Matrix4x4
    convention: Literal["opencv"] = "opencv"

    @model_validator(mode="after")
    def validate_homogeneous_extrinsic(self) -> CameraParameters:
        if not all(
            math.isclose(actual, expected, abs_tol=1e-8)
            for actual, expected in zip(
                self.world_to_camera[3], (0.0, 0.0, 0.0, 1.0), strict=True
            )
        ):
            raise ValueError(
                "world_to_camera must be homogeneous with last row [0, 0, 0, 1]."
            )
        return self

    @classmethod
    def from_matrices(
        cls,
        *,
        intrinsics: Sequence[Sequence[float]] | np.ndarray,
        world_to_camera: Sequence[Sequence[float]] | np.ndarray,
    ) -> CameraParameters:
        intrinsic_array = _matrix(intrinsics, (3, 3), "intrinsics")
        extrinsic_array = _matrix(world_to_camera, (4, 4), "world_to_camera")
        if not np.allclose(extrinsic_array[3], (0.0, 0.0, 0.0, 1.0)):
            raise InvalidRequestError(
                "world_to_camera must be homogeneous with last row [0, 0, 0, 1]."
            )
        return cls(
            intrinsics=tuple(tuple(float(value) for value in row) for row in intrinsic_array),
            world_to_camera=tuple(
                tuple(float(value) for value in row) for row in extrinsic_array
            ),
        )

    @classmethod
    def from_camera_to_world(
        cls,
        *,
        intrinsics: Sequence[Sequence[float]] | np.ndarray,
        camera_to_world: Sequence[Sequence[float]] | np.ndarray,
    ) -> CameraParameters:
        camera_to_world_array = _matrix(camera_to_world, (4, 4), "camera_to_world")
        try:
            world_to_camera = np.linalg.inv(camera_to_world_array)
        except np.linalg.LinAlgError as exc:
            raise InvalidRequestError("camera_to_world must be invertible.") from exc
        return cls.from_matrices(
            intrinsics=intrinsics,
            world_to_camera=world_to_camera,
        )

    @property
    def intrinsics_array(self) -> np.ndarray:
        return np.asarray(self.intrinsics, dtype=np.float32)

    @property
    def world_to_camera_array(self) -> np.ndarray:
        return np.asarray(self.world_to_camera, dtype=np.float32)


class VGGTOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    depth_conf_threshold: float | None = Field(default=None, ge=0.0)


class DepthAnything3Options(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    align_to_input_ext_scale: bool = True
    infer_gs: bool = False
    export_point_cloud: bool = True
    use_ray_pose: bool = False
    ref_view_strategy: Literal[
        "first", "middle", "saddle_balanced", "saddle_sim_range"
    ] = "saddle_balanced"
    process_res: int = Field(default=504, ge=224, le=2016)
    process_res_method: Literal["upper_bound_resize", "lower_bound_resize"] = (
        "upper_bound_resize"
    )


class BackendInputField(BaseModel):
    required: bool


class BackendDescriptor(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: str
    model_id: str
    model_revision: str | None = None
    inputs: dict[str, BackendInputField]
    outputs: list[str]
    options_schema: dict[str, Any]


class ReadyStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: Literal["ready", "not_ready"]
    ready: bool
    backend: str
    capabilities: list[str] = Field(default_factory=list)
    device: str | None = None
    error: str | None = None


class ImageSize(BaseModel):
    width: int
    height: int


class CameraResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    convention: Literal["opencv"]
    world_to_camera: Matrix4x4 | None = None
    intrinsics: Matrix3x3 | None = None
    source: Literal["predicted", "provided", "aligned"]

    @property
    def intrinsics_array(self) -> np.ndarray | None:
        return None if self.intrinsics is None else np.asarray(self.intrinsics, dtype=np.float32)

    @property
    def world_to_camera_array(self) -> np.ndarray | None:
        return (
            None
            if self.world_to_camera is None
            else np.asarray(self.world_to_camera, dtype=np.float32)
        )


class ViewResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    view_id: str
    filename: str
    original_size: ImageSize
    camera: CameraResult | None = None


class Artifact(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    kind: str
    url: str
    content_type: str
    size_bytes: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class TimingStats(BaseModel):
    model_config = ConfigDict(extra="allow")

    validation: int = 0
    inference: int = 0
    postprocess: int = 0
    total: int = 0


class ServiceError(BaseModel):
    code: str
    message: str


class ReconstructionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    service_version: str
    result_schema_version: str
    backend: str
    model: BackendDescriptor
    request_id: str
    client_request_id: str | None = None
    status: Literal["succeeded", "failed"]
    timings_ms: TimingStats
    view_results: list[ViewResult] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    produced_outputs: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: ServiceError | None = None

    @model_validator(mode="after")
    def validate_success(self) -> ReconstructionResult:
        if self.status == "succeeded" and self.error is not None:
            raise ValueError("A successful result cannot contain an error.")
        return self

    def artifact(self, *, name: str | None = None, kind: str | None = None) -> Artifact:
        if (name is None) == (kind is None):
            raise ValueError("Specify exactly one of name or kind.")
        matches = [
            artifact
            for artifact in self.artifacts
            if (name is not None and artifact.name == name)
            or (kind is not None and artifact.kind == kind)
        ]
        if not matches:
            selector = f"name={name!r}" if name is not None else f"kind={kind!r}"
            raise KeyError(f"No artifact with {selector}.")
        if len(matches) > 1:
            raise KeyError(f"More than one artifact has kind={kind!r}; select by name.")
        return matches[0]


ImagePath = str | Path
Options = Mapping[str, Any] | BaseModel | None


def options_payload(options: Options) -> dict[str, Any]:
    if options is None:
        return {}
    if isinstance(options, BaseModel):
        return options.model_dump(mode="json", exclude_none=True)
    return dict(options)
