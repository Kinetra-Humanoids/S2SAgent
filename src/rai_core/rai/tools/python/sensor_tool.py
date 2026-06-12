# Copyright (C) 2026 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import Any, Literal

import numpy as np
from langchain_core.tools import BaseTool, tool

from rai.messages import preprocess_image


SensorTransport = Literal["zmq", "http"]


@lru_cache(maxsize=8)
def _get_camera_client(
    transport: SensorTransport,
    server_ip: str,
    port: int,
    http_base_url: str,
):
    try:
        from rai.tools.python.camera.composed_camera import (
            ComposedCameraClientSensor,
            ComposedCameraHttpClient,
        )
    except ImportError as e:
        raise RuntimeError(
            "Camera sensor dependencies are missing. Install opencv-python, "
            "pyzmq, msgpack, and msgpack-numpy before enabling sensor tools."
        ) from e

    if transport == "http":
        if not http_base_url:
            raise RuntimeError("HTTP camera transport requires sensor_http_base_url.")
        return ComposedCameraHttpClient(http_base_url)
    return ComposedCameraClientSensor(server_ip=server_ip, port=port)


def _normalize_camera_name(camera_name: str | None) -> str | None:
    if camera_name is None:
        return None
    stripped = camera_name.strip()
    return stripped or None


def _summarize_image(name: str, image: np.ndarray, timestamp: float | None) -> str:
    shape = "x".join(str(value) for value in image.shape)
    if timestamp is None:
        return f"{name}: shape={shape}"
    age_ms = (time.time() - float(timestamp)) * 1000.0
    return f"{name}: shape={shape}, age={age_ms:.0f}ms"


def get_sensor_tools(
    *,
    transport: SensorTransport = "zmq",
    server_ip: str | None = None,
    port: int | None = None,
    http_base_url: str | None = None,
    default_camera: str | None = None,
    blocking: bool = True,
) -> list[BaseTool]:
    configured_transport = transport
    configured_server_ip = server_ip or os.getenv("SENSOR_SERVER_IP", "localhost")
    configured_port = int(port or os.getenv("SENSOR_SERVER_PORT", "5555"))
    configured_http_base_url = http_base_url or os.getenv("SENSOR_HTTP_BASE_URL", "")
    configured_default_camera = _normalize_camera_name(
        default_camera or os.getenv("SENSOR_DEFAULT_CAMERA", "")
    )
    configured_blocking = blocking

    def client():
        return _get_camera_client(
            configured_transport,
            configured_server_ip,
            configured_port,
            configured_http_base_url,
        )

    @tool(response_format="content_and_artifact")
    def sensor_camera_observe(
        camera_name: str = "",
        blocking: bool | None = None,
    ) -> tuple[str, dict[str, list[str]]]:
        """Read the latest robot camera image or images.

        Args:
            camera_name: Optional camera name to read, such as ego_view, head,
                left_wrist, or right_wrist. Leave empty to return all cameras.
            blocking: Whether to wait for a fresh frame. Defaults to the configured value.
        """
        selected_camera = _normalize_camera_name(camera_name) or configured_default_camera
        should_block = configured_blocking if blocking is None else bool(blocking)
        data = client().read(blocking=should_block)
        if data is None:
            raise RuntimeError("No camera frame received from the sensor server.")

        images = data.get("images", {})
        timestamps = data.get("timestamps", {})
        if selected_camera is not None:
            if selected_camera not in images:
                available = ", ".join(sorted(images.keys())) or "none"
                raise ValueError(
                    f"Camera {selected_camera!r} is not available. "
                    f"Available cameras: {available}"
                )
            images = {selected_camera: images[selected_camera]}

        artifact_images: list[str] = []
        summary_lines: list[str] = []
        for name, image in images.items():
            if image is None:
                summary_lines.append(f"{name}: no frame")
                continue
            if not isinstance(image, np.ndarray):
                summary_lines.append(f"{name}: unsupported frame type {type(image).__name__}")
                continue
            artifact_images.append(preprocess_image(image))
            summary_lines.append(_summarize_image(name, image, timestamps.get(name)))

        if not artifact_images:
            raise RuntimeError("Camera data was received, but no usable image frame was found.")

        return (
            "Camera observation received:\n" + "\n".join(summary_lines),
            {"images": artifact_images, "audios": []},
        )

    return [sensor_camera_observe]
