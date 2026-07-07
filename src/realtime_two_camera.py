from __future__ import annotations

import argparse
import glob
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import viser
from huggingface_hub import hf_hub_download
from hydra import compose, initialize_config_dir
from jaxtyping import install_import_hook

from src.config import load_typed_root_config
from src.runtime import configure_torch_runtime

with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.ply_export import export_ply
    from src.model.types import Gaussians


@dataclass
class TimedGaussians:
    gaussians: Gaussians
    extrinsics: Optional[torch.Tensor]
    intrinsics: torch.Tensor
    midpoint_color: Optional[torch.Tensor]
    midpoint_depth: Optional[torch.Tensor]
    virtual_t: float
    frame_ids: tuple[int, int]
    timestamp: float


@dataclass
class RenderCamera:
    extrinsics: torch.Tensor
    intrinsics: torch.Tensor


MODEL_PRESETS = {
    "re10k-spfv2": (
        "spfv2_viewsplat/re10k_eval",
        "re10k_spfv2_viewsplat.ckpt",
    ),
    "acid-spfv2": (
        "spfv2_viewsplat/acid",
        "acid_spfv2_viewsplat.ckpt",
    ),
    "re10k-spfv2l": (
        "spfv2l_viewsplat/re10k",
        "re10k_spfv2l_viewsplat.ckpt",
    ),
    "acid-spfv2l": (
        "spfv2l_viewsplat/acid",
        "acid_spfv2l_viewsplat.ckpt",
    ),
}


def list_cameras(max_index: int = 16) -> list[dict[str, object]]:
    candidates = {int(path.removeprefix("/dev/video")) for path in glob.glob("/dev/video*") if path.removeprefix("/dev/video").isdigit()}
    candidates.update(range(max_index))
    cameras: list[dict[str, object]] = []
    for index in sorted(candidates):
        cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        try:
            if not cap.isOpened():
                continue
            ok, _ = cap.read()
            if not ok:
                continue
            cameras.append(
                {
                    "index": index,
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                    "backend": cap.getBackendName(),
                }
            )
        finally:
            cap.release()
    return cameras


def print_cameras(max_index: int) -> None:
    cameras = list_cameras(max_index)
    if not cameras:
        print("No readable V4L2 cameras found.")
        return
    for camera in cameras:
        print(
            f"index={camera['index']} "
            f"size={camera['width']}x{camera['height']} "
            f"fps={camera['fps']:.2f} backend={camera['backend']}"
        )


class LatestFrameCamera:
    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self._frames: queue.Queue[tuple[int, float, torch.Tensor]] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._frame_id = 0
        self._error: Optional[BaseException] = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def latest(self, timeout: float = 1.0) -> tuple[int, float, torch.Tensor]:
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            if self._error is not None:
                raise RuntimeError(f"Camera {self.index} failed") from self._error
            raise

    def _run(self) -> None:
        cap = cv2.VideoCapture(self.index, cv2.CAP_V4L2)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.index}")

            while not self._stop.is_set():
                ok, frame_bgr = cap.read()
                if not ok:
                    time.sleep(0.002)
                    continue

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
                    frame_rgb = cv2.resize(
                        frame_rgb,
                        (self.width, self.height),
                        interpolation=cv2.INTER_AREA,
                    )

                tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).contiguous()
                self._put_latest(tensor)
        except BaseException as exc:
            self._error = exc
        finally:
            cap.release()

    def _put_latest(self, tensor: torch.Tensor) -> None:
        item = (self._frame_id, time.perf_counter(), tensor)
        self._frame_id += 1

        if self._frames.full():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
        self._frames.put_nowait(item)


class SyntheticFrameCamera(LatestFrameCamera):
    def __init__(self, index: int, width: int, height: int, fps: int) -> None:
        super().__init__(index, width, height, fps)
        y = torch.linspace(0, 255, height, dtype=torch.uint8).view(1, height, 1)
        x = torch.linspace(0, 255, width, dtype=torch.uint8).view(1, 1, width)
        phase = torch.tensor(index * 37, dtype=torch.uint8)
        self._base = torch.cat(
            (
                x.expand(1, height, width),
                y.expand(1, height, width),
                (x + y + phase).expand(1, height, width),
            ),
            dim=0,
        ).contiguous()

    def _run(self) -> None:
        period = 1.0 / self.fps if self.fps > 0 else 0.0
        while not self._stop.is_set():
            self._put_latest(torch.roll(self._base, shifts=self._frame_id, dims=2))
            if period > 0:
                time.sleep(period)


class LatestImageViewer:
    def __init__(
        self,
        title: str,
        width: int,
        height: int,
        num_buffers: int = 3,
    ) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.display_scale = 3
        self._frames: queue.Queue[tuple[int, torch.cuda.Event]] = queue.Queue(maxsize=1)
        self._buffers = [
            torch.empty((height, width, 3), dtype=torch.uint8, pin_memory=True)
            for _ in range(num_buffers)
        ]
        self._copy_stream = torch.cuda.Stream()
        self._next_buffer = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass

    def submit(self, frame_rgb: torch.Tensor) -> None:
        buffer_index = self._next_buffer
        self._next_buffer = (self._next_buffer + 1) % len(self._buffers)
        buffer = self._buffers[buffer_index]

        with torch.cuda.stream(self._copy_stream):
            image = frame_rgb.detach().clamp(0, 1).mul(255).to(torch.uint8)
            image = image.permute(1, 2, 0).contiguous()
            buffer.copy_(image, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._copy_stream)

        if self._frames.full():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
        self._frames.put_nowait((buffer_index, event))

    def _run(self) -> None:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.title, self.width * self.display_scale, self.height * self.display_scale)
        while not self._stop.is_set():
            try:
                buffer_index, event = self._frames.get(timeout=0.02)
            except queue.Empty:
                cv2.waitKey(1)
                continue

            event.synchronize()
            frame_rgb = self._buffers[buffer_index].numpy()
            cv2.imshow(self.title, self._display_image(frame_rgb))
            if cv2.waitKey(1) & 0xFF == 27:
                self._stop.set()

    def _display_image(self, frame_rgb: np.ndarray) -> np.ndarray:
        try:
            _, _, window_width, window_height = cv2.getWindowImageRect(self.title)
        except cv2.error:
            window_width = self.width * self.display_scale
            window_height = self.height * self.display_scale

        window_width = max(self.width, window_width)
        window_height = max(self.height, window_height)
        scale = max(1, int(min(window_width / self.width, window_height / self.height)))
        display_width = self.width * scale
        display_height = self.height * scale
        if display_width == self.width and display_height == self.height:
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(
            frame_rgb,
            (display_width, display_height),
            interpolation=cv2.INTER_NEAREST,
        )
        return cv2.cvtColor(resized, cv2.COLOR_RGB2BGR)


class RawCameraViewer:
    def __init__(self, title: str, width: int, height: int) -> None:
        self.title = title
        self.width = width
        self.height = height
        self.display_scale = 2
        self._frames: queue.Queue[tuple[torch.Tensor, torch.Tensor]] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        try:
            cv2.destroyWindow(self.title)
        except cv2.error:
            pass

    def submit(self, left_rgb: torch.Tensor, right_rgb: torch.Tensor) -> None:
        if self._frames.full():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
        self._frames.put_nowait((left_rgb, right_rgb))

    def _run(self) -> None:
        cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            self.title,
            self.width * 2 * self.display_scale,
            self.height * self.display_scale,
        )
        while not self._stop.is_set():
            try:
                left_rgb, right_rgb = self._frames.get(timeout=0.02)
            except queue.Empty:
                cv2.waitKey(1)
                continue

            frame_rgb = np.concatenate(
                (
                    self._tensor_to_hwc(left_rgb),
                    self._tensor_to_hwc(right_rgb),
                ),
                axis=1,
            )
            cv2.imshow(self.title, self._display_image(frame_rgb))
            if cv2.waitKey(1) & 0xFF == 27:
                self._stop.set()

    def _tensor_to_hwc(self, frame_rgb: torch.Tensor) -> np.ndarray:
        if frame_rgb.is_cuda:
            frame_rgb = frame_rgb.cpu()
        frame = frame_rgb.detach().permute(1, 2, 0).numpy()
        return np.ascontiguousarray(frame)

    def _display_image(self, frame_rgb: np.ndarray) -> np.ndarray:
        base_width = self.width * 2
        try:
            _, _, window_width, window_height = cv2.getWindowImageRect(self.title)
        except cv2.error:
            window_width = base_width * self.display_scale
            window_height = self.height * self.display_scale

        window_width = max(base_width, window_width)
        window_height = max(self.height, window_height)
        scale = max(1, int(min(window_width / base_width, window_height / self.height)))
        display_width = base_width * scale
        display_height = self.height * scale
        if display_width != base_width or display_height != self.height:
            frame_rgb = cv2.resize(
                frame_rgb,
                (display_width, display_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


class ViserImageStreamer:
    def __init__(
        self,
        width: int,
        height: int,
        port: int,
        jpeg_quality: int,
        max_fps: float,
        scene_fps: float,
        stream_scale: int,
    ) -> None:
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.stream_scale = max(1, stream_scale)
        self.min_period = 1.0 / max_fps if max_fps > 0 else 0.0
        self.min_scene_period = 1.0 / scene_fps if scene_fps > 0 else None
        self._last_send = 0.0
        self._last_scene_update = 0.0
        self._scene_initialized = False
        self._latest_camera: Optional[RenderCamera] = None
        self._default_camera: Optional[RenderCamera] = None
        self._clients: dict[int, viser.ClientHandle] = {}
        self._client_initialized: set[int] = set()
        self._lock = threading.Lock()
        self._frames: queue.Queue[tuple[int, torch.cuda.Event]] = queue.Queue(maxsize=1)
        self._host_buffers = [
            torch.empty((height, width, 3), dtype=torch.uint8, pin_memory=True)
            for _ in range(3)
        ]
        self._next_buffer = 0
        self._copy_stream = torch.cuda.Stream()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.sent_frames = 0
        self.server = viser.ViserServer(port=port)
        self.server.scene.set_up_direction("-y")
        self.server.scene.world_axes.visible = True
        self.left_frustum = self.server.scene.add_camera_frustum(
            "/cameras/left",
            fov=0.9,
            aspect=width / height,
            scale=0.12,
            color=(255, 80, 80),
            visible=True,
        )
        self.right_frustum = self.server.scene.add_camera_frustum(
            "/cameras/right",
            fov=0.9,
            aspect=width / height,
            scale=0.12,
            color=(80, 160, 255),
            visible=True,
        )
        self.virtual_frustum = self.server.scene.add_camera_frustum(
            "/cameras/render",
            fov=0.9,
            aspect=width / height,
            scale=0.16,
            color=(80, 255, 120),
            visible=True,
        )
        print(f"viser: http://localhost:{port}", flush=True)

        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            with self._lock:
                self._clients[client.client_id] = client
            client.camera.near = 0.01
            client.camera.far = 100.0
            self._initialize_client_camera(client)

            @client.camera.on_update
            def _(_camera: viser.CameraHandle) -> None:
                self._update_camera(client)

        @self.server.on_client_disconnect
        def _(client: viser.ClientHandle) -> None:
            with self._lock:
                self._clients.pop(client.client_id, None)
                self._client_initialized.discard(client.client_id)

        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def latest_camera(self, device: torch.device) -> Optional[RenderCamera]:
        with self._lock:
            camera = self._latest_camera
        if camera is None:
            return None
        return RenderCamera(
            camera.extrinsics.to(device, non_blocking=True),
            camera.intrinsics.to(device, non_blocking=True),
        )

    def update_scene(self, output: TimedGaussians) -> None:
        if output.extrinsics is None:
            return
        now = time.perf_counter()
        if (
            self.min_scene_period is not None
            and now - self._last_scene_update < self.min_scene_period
        ):
            return
        self._last_scene_update = now
        self._scene_initialized = True

        left = output.extrinsics[:, 0].detach()
        right = output.extrinsics[:, 1].detach()
        virtual = virtual_camera_extrinsics(left, right, output.virtual_t).detach()
        intrinsics = output.intrinsics.detach()

        self._update_frustum(self.left_frustum, left[0], intrinsics[0, 0])
        self._update_frustum(self.right_frustum, right[0], intrinsics[0, 1])
        self._update_frustum(self.virtual_frustum, virtual[0], (intrinsics[0, 0] + intrinsics[0, 1]) * 0.5)

        default_camera = RenderCamera(
            virtual.unsqueeze(1).cpu().pin_memory(),
            ((intrinsics[:, 0] + intrinsics[:, 1]) * 0.5).unsqueeze(1).cpu().pin_memory(),
        )
        with self._lock:
            self._default_camera = default_camera
            clients = list(self._clients.values())
        for client in clients:
            self._initialize_client_camera(client)

    def _update_frustum(self, handle: viser.CameraFrustumHandle, extrinsics: torch.Tensor, intrinsics: torch.Tensor) -> None:
        wxyz, position = opencv_c2w_to_viser_pose(extrinsics)
        handle.wxyz = wxyz
        handle.position = position

    def submit(self, frame_rgb: torch.Tensor) -> None:
        now = time.perf_counter()
        if self.min_period > 0 and now - self._last_send < self.min_period:
            return
        self._last_send = now
        buffer_index = self._next_buffer
        self._next_buffer = (self._next_buffer + 1) % len(self._host_buffers)
        host_buffer = self._host_buffers[buffer_index]

        with torch.cuda.stream(self._copy_stream):
            image = frame_rgb.detach().clamp(0, 1).mul(255).to(torch.uint8)
            image = image.permute(1, 2, 0).contiguous()
            host_buffer.copy_(image, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._copy_stream)

        if self._frames.full():
            try:
                self._frames.get_nowait()
            except queue.Empty:
                pass
        self._frames.put_nowait((buffer_index, event))

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                buffer_index, event = self._frames.get(timeout=0.05)
            except queue.Empty:
                continue
            event.synchronize()
            self.server.scene.set_background_image(
                self._stream_image(self._host_buffers[buffer_index].numpy()),
                format="jpeg",
                jpeg_quality=self.jpeg_quality,
            )
            self.sent_frames += 1

    def _stream_image(self, image: np.ndarray) -> np.ndarray:
        if self.stream_scale == 1:
            return image
        return cv2.resize(
            image,
            (self.width * self.stream_scale, self.height * self.stream_scale),
            interpolation=cv2.INTER_CUBIC,
        )

    def _update_camera(self, client: viser.ClientHandle) -> None:
        with self._lock:
            if client.client_id not in self._client_initialized:
                return
        camera = client.camera
        if camera.image_width <= 0 or camera.image_height <= 0:
            return

        wxyz = torch.tensor(camera.wxyz, dtype=torch.float32).view(1, 4)
        position = torch.tensor(camera.position, dtype=torch.float32).view(1, 3)
        rotation = quaternion_to_rotation_matrix(wxyz)

        extrinsics = torch.eye(4, dtype=torch.float32).view(1, 4, 4)
        extrinsics[:, :3, :3] = rotation
        extrinsics[:, :3, 3] = position
        extrinsics = extrinsics.unsqueeze(1).pin_memory()

        fov_y = float(camera.fov)
        aspect = float(camera.aspect)
        fy = 0.5 / np.tan(0.5 * fov_y)
        fx = fy / aspect
        intrinsics = torch.tensor(
            [
                [fx, 0.0, 0.5],
                [0.0, fy, 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ).view(1, 1, 3, 3).pin_memory()

        with self._lock:
            self._latest_camera = RenderCamera(extrinsics, intrinsics)

    def _initialize_client_camera(self, client: viser.ClientHandle) -> None:
        with self._lock:
            if client.client_id in self._client_initialized or self._default_camera is None:
                return
            default_camera = self._default_camera
            self._client_initialized.add(client.client_id)

        extrinsics = default_camera.extrinsics[0, 0]
        rotation = extrinsics[:3, :3].detach().cpu().numpy().astype(np.float64)
        position = extrinsics[:3, 3].detach().cpu().numpy().astype(np.float64)
        client.camera.position = position
        client.camera.look_at = position + rotation[:, 2]
        client.camera.up_direction = -rotation[:, 1]


class TwoCameraViewSplatPipeline:
    def __init__(
        self,
        checkpoint: Path,
        experiment: str,
        disable_view_dependent_head: bool,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        compile_heads: bool,
        compile_encoder: bool,
        compile_mode: str,
        half_encoder: bool,
        channels_last: bool,
        render_midpoint: bool,
        near: float,
        far: float,
        device: str = "cuda",
    ) -> None:
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the realtime pipeline.")

        self.device = torch.device(device)
        self.width = width
        self.height = height
        self.transfer_stream = torch.cuda.Stream(device=self.device)
        self.host_pair = torch.empty((2, 3, height, width), dtype=torch.uint8, pin_memory=True)
        self.device_pair = torch.empty(
            (2, 3, height, width),
            device=self.device,
            dtype=torch.float16,
            memory_format=torch.channels_last,
        )
        self.intrinsics = self._make_intrinsics(fx, fy, cx, cy).to(self.device)
        self.extrinsics = torch.eye(4, dtype=torch.float32, device=self.device).repeat(1, 2, 1, 1)
        self.near = torch.full((1, 1), near, dtype=torch.float32, device=self.device)
        self.far = torch.full((1, 1), far, dtype=torch.float32, device=self.device)
        self.render_midpoint = render_midpoint
        self.experiment = experiment
        self.disable_view_dependent_head = disable_view_dependent_head
        self.encoder = self._load_encoder(
            checkpoint=checkpoint,
            experiment=experiment,
            disable_view_dependent_head=disable_view_dependent_head,
            compile_heads=compile_heads,
            compile_encoder=compile_encoder,
            compile_mode=compile_mode,
            half_encoder=half_encoder,
            channels_last=channels_last,
        )
        self.renderer = self._load_renderer().to(self.device).eval() if render_midpoint else None

    def infer(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        frame_ids: tuple[int, int],
        virtual_t: float,
        render_camera: Optional[RenderCamera] = None,
    ) -> TimedGaussians:
        self.host_pair[0].copy_(left)
        self.host_pair[1].copy_(right)
        with torch.cuda.stream(self.transfer_stream):
            self.device_pair.copy_(self.host_pair, non_blocking=True)
            self.device_pair.div_(255.0)
            image = self.device_pair.unsqueeze(0)
            context = {
                "image": image,
                "intrinsics": self.intrinsics,
                "extrinsics": self.extrinsics,
            }

        torch.cuda.current_stream(self.device).wait_stream(self.transfer_stream)
        with torch.inference_mode():
            output = self.encoder(context, global_step=0, visualization_dump=None)
            extrinsics = output.get("extrinsics", {}).get("c")
            midpoint_color, midpoint_depth = self._render_midpoint(
                output["gaussians"],
                extrinsics,
                virtual_t,
                render_camera,
            )

        return TimedGaussians(
            gaussians=output["gaussians"],
            extrinsics=extrinsics,
            intrinsics=self.intrinsics,
            midpoint_color=midpoint_color,
            midpoint_depth=midpoint_depth,
            virtual_t=virtual_t,
            frame_ids=frame_ids,
            timestamp=time.perf_counter(),
        )

    def _make_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> torch.Tensor:
        intrinsics = torch.tensor(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
        )
        return intrinsics.view(1, 1, 3, 3).repeat(1, 2, 1, 1)

    def _render_midpoint(
        self,
        gaussians: Gaussians,
        extrinsics: Optional[torch.Tensor],
        virtual_t: float,
        render_camera: Optional[RenderCamera],
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self.renderer is None:
            return None, None
        if render_camera is not None:
            rendered = self.renderer.forward(
                gaussians,
                render_camera.extrinsics,
                render_camera.intrinsics,
                self.near,
                self.far,
                (self.height, self.width),
            )
            return rendered.color, rendered.depth
        if extrinsics is None:
            raise RuntimeError("Midpoint rendering requires estimated extrinsics.")

        midpoint_extrinsics = virtual_camera_extrinsics(
            extrinsics[:, 0],
            extrinsics[:, 1],
            virtual_t,
        ).unsqueeze(1)
        midpoint_intrinsics = (
            self.intrinsics[:, 0] + (self.intrinsics[:, 1] - self.intrinsics[:, 0]) * virtual_t
        ).unsqueeze(1)
        rendered = self.renderer.forward(
            gaussians,
            midpoint_extrinsics,
            midpoint_intrinsics,
            self.near,
            self.far,
            (self.height, self.width),
        )
        return rendered.color, rendered.depth

    def _load_encoder(
        self,
        checkpoint: Path,
        experiment: str,
        disable_view_dependent_head: bool,
        compile_heads: bool,
        compile_encoder: bool,
        compile_mode: str,
        half_encoder: bool,
        channels_last: bool,
    ) -> torch.nn.Module:
        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(repo_root / "config")):
            overrides = [
                f"+experiment={experiment}",
                f"checkpointing.load={checkpoint}",
                "mode=test",
                f"test.compile_heads={str(compile_heads).lower()}",
                f"test.compile_encoder={str(compile_encoder).lower()}",
                f"test.compile_mode={compile_mode}",
                f"test.half_encoder={str(half_encoder).lower()}",
                f"test.channels_last={str(channels_last).lower()}",
            ]
            if disable_view_dependent_head:
                overrides.append("model.encoder.skip_view_dependent_head_in_fast_inference=true")
            cfg_dict = compose(
                config_name="main",
                overrides=overrides,
            )
        cfg = load_typed_root_config(cfg_dict)
        encoder, _ = get_encoder(cfg.model.encoder)

        ckpt = torch.load(checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        if any(key.startswith("encoder.") for key in state_dict):
            state_dict = {
                key.removeprefix("encoder."): value
                for key, value in state_dict.items()
                if key.startswith("encoder.")
            }
        missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Checkpoint load mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
            )

        encoder = encoder.eval().to(self.device)
        if channels_last and encoder.cfg.name != "spfv2l_viewsplat":
            encoder = encoder.to(memory_format=torch.channels_last)
        if half_encoder and encoder.cfg.name != "spfv2l_viewsplat":
            encoder = encoder.half()

        if compile_heads:
            for name in (
                "downstream_head1",
                "downstream_head2",
                "gaussian_param_head",
                "gaussian_param_head2",
            ):
                if hasattr(encoder, name):
                    setattr(
                        encoder,
                        name,
                        torch.compile(
                            getattr(encoder, name),
                            mode=compile_mode,
                            fullgraph=False,
                        ),
                    )

        if compile_encoder:
            encoder = torch.compile(encoder, mode=compile_mode, fullgraph=False)

        return encoder

    def _load_renderer(self) -> torch.nn.Module:
        repo_root = Path(__file__).resolve().parents[1]
        with initialize_config_dir(version_base=None, config_dir=str(repo_root / "config")):
            cfg_dict = compose(
                config_name="main",
                overrides=[
                    f"+experiment={self.experiment}",
                    "mode=test",
                ],
            )
        cfg = load_typed_root_config(cfg_dict)
        return get_decoder(cfg.model.decoder)


def virtual_camera_extrinsics(left: torch.Tensor, right: torch.Tensor, t: float) -> torch.Tensor:
    midpoint = torch.empty_like(left)
    midpoint[:, :3, :3] = slerp_rotation_matrices(left[:, :3, :3], right[:, :3, :3], t)
    midpoint[:, :3, 3] = left[:, :3, 3] + (right[:, :3, 3] - left[:, :3, 3]) * t
    midpoint[:, 3, :3] = 0
    midpoint[:, 3, 3] = 1
    return midpoint


def opencv_c2w_to_viser_pose(extrinsics: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    wxyz = normalize_quaternion(rotation_matrix_to_quaternion(extrinsics[:3, :3].unsqueeze(0)))[0]
    position = extrinsics[:3, 3]
    return (
        wxyz.detach().cpu().numpy().astype(np.float64),
        position.detach().cpu().numpy().astype(np.float64),
    )


def slerp_rotation_matrices(left: torch.Tensor, right: torch.Tensor, t: float) -> torch.Tensor:
    q0 = normalize_quaternion(rotation_matrix_to_quaternion(left))
    q1 = normalize_quaternion(rotation_matrix_to_quaternion(right))
    dot = (q0 * q1).sum(dim=-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    dot = dot.abs().clamp(max=0.9995)

    linear = normalize_quaternion(q0 + (q1 - q0) * t)
    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * t
    s0 = torch.sin(theta_0 - theta) / sin_theta_0
    s1 = torch.sin(theta) / sin_theta_0
    spherical = s0 * q0 + s1 * q1
    use_linear = dot > 0.999
    return quaternion_to_rotation_matrix(torch.where(use_linear, linear, spherical))


def normalize_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    return quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    m00 = matrix[:, 0, 0]
    m11 = matrix[:, 1, 1]
    m22 = matrix[:, 2, 2]
    qw = torch.sqrt((1 + m00 + m11 + m22).clamp_min(1e-8)) * 0.5
    qx = torch.copysign(
        torch.sqrt((1 + m00 - m11 - m22).clamp_min(1e-8)) * 0.5,
        matrix[:, 2, 1] - matrix[:, 1, 2],
    )
    qy = torch.copysign(
        torch.sqrt((1 - m00 + m11 - m22).clamp_min(1e-8)) * 0.5,
        matrix[:, 0, 2] - matrix[:, 2, 0],
    )
    qz = torch.copysign(
        torch.sqrt((1 - m00 - m11 + m22).clamp_min(1e-8)) * 0.5,
        matrix[:, 1, 0] - matrix[:, 0, 1],
    )
    return torch.stack((qw, qx, qy, qz), dim=-1)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    qw, qx, qy, qz = normalize_quaternion(quaternion).unbind(dim=-1)
    two = 2.0
    matrix = torch.empty((*quaternion.shape[:-1], 3, 3), dtype=quaternion.dtype, device=quaternion.device)
    matrix[:, 0, 0] = 1 - two * (qy * qy + qz * qz)
    matrix[:, 0, 1] = two * (qx * qy - qz * qw)
    matrix[:, 0, 2] = two * (qx * qz + qy * qw)
    matrix[:, 1, 0] = two * (qx * qy + qz * qw)
    matrix[:, 1, 1] = 1 - two * (qx * qx + qz * qz)
    matrix[:, 1, 2] = two * (qy * qz - qx * qw)
    matrix[:, 2, 0] = two * (qx * qz - qy * qw)
    matrix[:, 2, 1] = two * (qy * qz + qx * qw)
    matrix[:, 2, 2] = 1 - two * (qx * qx + qy * qy)
    return matrix


def save_latest(output: TimedGaussians, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "frame_ids": output.frame_ids,
            "timestamp": output.timestamp,
            "gaussians": output.gaussians,
            "extrinsics": output.extrinsics,
            "intrinsics": output.intrinsics,
            "midpoint_color": output.midpoint_color,
            "midpoint_depth": output.midpoint_depth,
            "virtual_t": output.virtual_t,
        },
        path,
    )


def save_ply(output: TimedGaussians, path: Path) -> None:
    extrinsics = (
        output.extrinsics[0, 0]
        if output.extrinsics is not None
        else torch.eye(4, dtype=torch.float32, device=output.gaussians.means.device)
    )
    gaussians = output.gaussians
    export_ply(
        extrinsics,
        gaussians.means[0],
        gaussians.scales[0],
        gaussians.rotations[0],
        gaussians.harmonics[0],
        gaussians.opacities[0],
        path,
    )


def save_render(output: TimedGaussians, path: Path) -> None:
    if output.midpoint_color is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), midpoint_render_bgr(output))


def midpoint_render_bgr(output: TimedGaussians) -> Optional[np.ndarray]:
    if output.midpoint_color is None:
        return None
    image = output.midpoint_color[0, 0].detach().clamp(0, 1)
    image = image.permute(1, 2, 0).mul(255).byte().cpu().numpy()
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def resolve_model_selection(args: argparse.Namespace) -> tuple[str, Path]:
    if args.experiment is not None:
        experiment = args.experiment
        checkpoint = args.checkpoint
        if checkpoint is None:
            raise ValueError("--checkpoint is required when --experiment is specified.")
        return experiment, checkpoint

    experiment, checkpoint_name = MODEL_PRESETS[args.preset]
    checkpoint = args.checkpoint or (Path("pretrained_weights") / checkpoint_name)
    if not checkpoint.exists():
        if not args.download_checkpoint:
            raise FileNotFoundError(
                f"{checkpoint} does not exist. Re-run without --no-download-checkpoint or download it manually."
            )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id="myeon01/ViewSplat",
            filename=checkpoint_name,
            local_dir=checkpoint.parent,
        )
    return experiment, checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime 2-camera ViewSplat 3DGS pipeline.")
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--camera-scan-max-index", type=int, default=16)
    parser.add_argument("--left-camera", type=int, default=0)
    parser.add_argument("--right-camera", type=int, default=1)
    parser.add_argument("--preset", choices=sorted(MODEL_PRESETS), default="re10k-spfv2")
    parser.add_argument("--experiment", default=None, help="Hydra experiment override, e.g. spfv2l_viewsplat/acid.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--download-checkpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/realtime_two_camera"))
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=60)
    parser.add_argument("--fx", type=float, default=0.9, help="Normalized focal length x.")
    parser.add_argument("--fy", type=float, default=0.9, help="Normalized focal length y.")
    parser.add_argument("--cx", type=float, default=0.5, help="Normalized principal point x.")
    parser.add_argument("--cy", type=float, default=0.5, help="Normalized principal point y.")
    parser.add_argument("--save-every", type=int, default=0, help="Save latest 3DGS .pt every N frames. 0 disables disk output.")
    parser.add_argument("--save-ply-every", type=int, default=0, help="Save PLY every N frames. This is slow; keep it low-frequency.")
    parser.add_argument("--save-render-every", type=int, default=0, help="Save the midpoint render PNG every N frames. 0 disables disk output.")
    parser.add_argument("--show", action="store_true", help="Show the midpoint render in an OpenCV window.")
    parser.add_argument("--show-cameras", action="store_true", help="Show raw left/right camera frames in an OpenCV window.")
    parser.add_argument("--viser", action="store_true", help="Stream CUDA-rendered images to a viser browser view.")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--viser-max-fps", type=float, default=0.0, help="0 streams every rendered frame.")
    parser.add_argument("--viser-scene-fps", type=float, default=20.0, help="0 updates frustums every rendered frame. Image streaming is controlled separately.")
    parser.add_argument("--viser-jpeg-quality", type=int, default=75)
    parser.add_argument("--viser-stream-scale", type=int, default=2, help="CPU upscaling factor before JPEG streaming to viser.")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--synthetic", action="store_true", help="Use generated frames instead of physical cameras.")
    parser.add_argument("--render-midpoint", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--virtual-t", type=float, default=0.5, help="Virtual camera position: 0=left, 1=right.")
    parser.add_argument("--sweep", action="store_true", help="Sweep the virtual camera between the two estimated cameras.")
    parser.add_argument("--sweep-period", type=float, default=1.0, help="Seconds for one left-to-right-to-left sweep.")
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--compile-heads", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile-encoder", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", default="default")
    parser.add_argument("--half-encoder", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--disable-view-dependent-head", action="store_true")
    return parser.parse_args()


def sweep_position(elapsed: float, period: float) -> float:
    if period <= 0:
        return 0.5
    phase = (elapsed % period) / period
    return 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        print_cameras(args.camera_scan_max_index)
        return

    configure_torch_runtime()
    experiment, checkpoint = resolve_model_selection(args)
    if args.width is None:
        args.width = 224 if "spfv2l" in args.preset else 256
    if args.height is None:
        args.height = 224 if "spfv2l" in args.preset else 256
    if "spfv2l" in args.preset and (args.width % 14 != 0 or args.height % 14 != 0):
        raise ValueError("spfv2l presets require --width and --height to be multiples of 14.")
    print(f"model: preset={args.preset} experiment={experiment} checkpoint={checkpoint}", flush=True)

    pipeline = TwoCameraViewSplatPipeline(
        checkpoint=checkpoint,
        experiment=experiment,
        disable_view_dependent_head=args.disable_view_dependent_head,
        width=args.width,
        height=args.height,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        compile_heads=args.compile_heads,
        compile_encoder=args.compile_encoder,
        compile_mode=args.compile_mode,
        half_encoder=args.half_encoder,
        channels_last=args.channels_last,
        render_midpoint=args.render_midpoint,
        near=args.near,
        far=args.far,
    )

    camera_cls = SyntheticFrameCamera if args.synthetic else LatestFrameCamera
    left_camera = camera_cls(args.left_camera, args.width, args.height, args.camera_fps)
    right_camera = camera_cls(args.right_camera, args.width, args.height, args.camera_fps)
    left_camera.start()
    right_camera.start()
    viewer = (
        LatestImageViewer(
            "ViewSplat midpoint render",
            args.width,
            args.height,
        )
        if args.show
        else None
    )
    if viewer is not None:
        viewer.start()
    camera_viewer = (
        RawCameraViewer(
            "ViewSplat raw cameras",
            args.width,
            args.height,
        )
        if args.show_cameras
        else None
    )
    if camera_viewer is not None:
        camera_viewer.start()
    viser_streamer = (
        ViserImageStreamer(
            args.width,
            args.height,
            args.viser_port,
            args.viser_jpeg_quality,
            args.viser_max_fps,
            args.viser_scene_fps,
            args.viser_stream_scale,
        )
        if args.viser
        else None
    )

    frame_count = 0
    report_start = time.perf_counter()

    try:
        while args.max_frames <= 0 or frame_count < args.max_frames:
            left_id, _, left = left_camera.latest()
            right_id, _, right = right_camera.latest()
            if camera_viewer is not None:
                camera_viewer.submit(left, right)

            start = time.perf_counter()
            virtual_t = sweep_position(start - report_start, args.sweep_period) if args.sweep else args.virtual_t
            render_camera = viser_streamer.latest_camera(pipeline.device) if viser_streamer is not None else None
            output = pipeline.infer(left, right, (left_id, right_id), virtual_t, render_camera)
            torch.cuda.synchronize()
            infer_end = time.perf_counter()

            if args.warmup_frames > 0:
                args.warmup_frames -= 1
                report_start = infer_end
                continue

            frame_count += 1
            if args.save_every > 0 and frame_count % args.save_every == 0:
                save_latest(output, args.output_dir / "latest_gaussians.pt")
            if args.save_ply_every > 0 and frame_count % args.save_ply_every == 0:
                save_ply(output, args.output_dir / f"gaussians_{frame_count:06d}.ply")
            if args.save_render_every > 0 and frame_count % args.save_render_every == 0:
                save_render(output, args.output_dir / "midpoint.png")
            if args.show:
                if output.midpoint_color is not None:
                    viewer.submit(output.midpoint_color[0, 0])
            if viser_streamer is not None and output.midpoint_color is not None:
                viser_streamer.update_scene(output)
                viser_streamer.submit(output.midpoint_color[0, 0])

            if frame_count == 1 or frame_count % 20 == 0:
                window = infer_end - report_start
                fps = frame_count / window if window > 0 else 0.0
                infer_ms = (infer_end - start) * 1000.0
                print(
                    f"frames={frame_count} fps={fps:.2f} infer_ms={infer_ms:.2f} "
                    f"camera_frames=({left_id},{right_id}) gaussians={output.gaussians.means.shape[1]} "
                    f"render={output.midpoint_color is not None} virtual_t={output.virtual_t:.3f} "
                    f"viser_sent={viser_streamer.sent_frames if viser_streamer is not None else 0}",
                    flush=True,
                )
    finally:
        left_camera.stop()
        right_camera.stop()
        if viewer is not None:
            viewer.stop()
        if camera_viewer is not None:
            camera_viewer.stop()
    if viser_streamer is not None:
        viser_streamer.stop()


if __name__ == "__main__":
    main()
