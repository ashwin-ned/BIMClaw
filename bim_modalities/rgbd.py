"""RGBDLoader: BIM-Walker depth+poses → BIMReconstruction.

BIM-Walker outputs:
  - rgb/*.png            — RGB frames
  - depth_exr/*.exr      — metric depth (float32, metres, same HW as RGB)
  - pose/poses.txt       — one 4×4 camera-to-world matrix per frame
  - camera_info.json     — {"fx", "fy", "cx", "cy", "width", "height"}

BIMReconstruction wraps these with an API identical to SpatialClaw's
Reconstruction type (.depth / .extrinsics / .intrinsics / .points) so the
BIM agent can reuse SpatialClaw's geometry and BEV drawing utilities without
any GPU reconstruction step.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _load_exr_depth(exr_path: Path) -> np.ndarray:
    """Load a depth EXR file as float32 (H, W) in metres.

    Tries imageio (with freeimage plugin) first; falls back to OpenCV.
    """
    path = Path(exr_path)
    try:
        import imageio
        arr = imageio.v3.imread(str(path), plugin="freeimage")
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr.astype(np.float32)
    except Exception:
        pass

    try:
        import cv2
        arr = cv2.imread(str(path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if arr is None:
            raise IOError(f"cv2 could not read {path}")
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr.astype(np.float32)
    except Exception as exc:
        raise IOError(
            f"Cannot load depth EXR {path}. "
            f"Install imageio[freeimage] or opencv-python. Error: {exc}"
        ) from exc


def _parse_poses(pose_file: Path) -> List[np.ndarray]:
    """Parse BIM-Walker poses.txt → list of 4×4 camera-to-world matrices.

    Format (one block per frame, separated by blank lines):
        frame_0000
        r00 r01 r02 tx
        r10 r11 r12 ty
        r20 r21 r22 tz
        0   0   0   1
    """
    text = Path(pose_file).read_text()
    blocks = [b.strip() for b in text.strip().split("\n\n") if b.strip()]
    matrices: List[np.ndarray] = []
    for block in blocks:
        lines = block.strip().splitlines()
        # Skip optional label line (starts with non-digit / non-minus after whitespace)
        data_lines = [l for l in lines if l.strip() and l.strip()[0] in "0123456789-+"]
        if len(data_lines) < 4:
            continue
        rows = []
        for l in data_lines[:4]:
            rows.append([float(v) for v in l.split()])
        matrices.append(np.array(rows, dtype=np.float64))
    return matrices


@dataclass
class BIMFrame:
    """One rendered frame from BIM-Walker with metric depth and camera pose."""

    frame_index: int
    rgb: Image.Image
    depth: np.ndarray                    # (H, W) float32 in metres; 0 = invalid
    camera_to_world: np.ndarray          # (4, 4) float64
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def intrinsics(self) -> Dict[str, float]:
        return {"fx": self.fx, "fy": self.fy, "cx": self.cx, "cy": self.cy}

    def unproject(self) -> np.ndarray:
        """Unproject depth image → (H, W, 3) world-space XYZ float32.

        Invalid depth pixels (depth == 0 or inf) produce NaN XYZ.
        """
        H, W = self.depth.shape
        u = np.arange(W, dtype=np.float32)
        v = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        valid = (self.depth > 0) & np.isfinite(self.depth)
        d = self.depth.copy()
        d[~valid] = np.nan

        x_cam = (uu - self.cx) / self.fx * d
        y_cam = (vv - self.cy) / self.fy * d
        z_cam = d

        pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(d)], axis=-1)  # H×W×4
        pts_world = (self.camera_to_world @ pts_cam.reshape(-1, 4).T).T
        return pts_world[:, :3].reshape(H, W, 3).astype(np.float32)


class BIMReconstruction:
    """Reconstruction built from BIM-Walker RGBD output.

    API is intentionally identical to SpatialClaw's ``Reconstruction`` type
    so that geometry tools and BEV rendering can be reused without changes.

    Attributes (matching SpatialClaw Reconstruction)
    ---------
    depth[i]:
        (H, W) float32 depth in metres for absolute frame index i.
    extrinsics[i]:
        (4, 4) camera-to-world matrix for frame i.
    intrinsics[i]:
        dict {fx, fy, cx, cy} for frame i.
    points[i]:
        (H, W, 3) float32 world-space XYZ (NaN where depth invalid).
    frames:
        List of BIMFrame objects.
    """

    def __init__(self, frames: List[BIMFrame]):
        self.frames: List[BIMFrame] = frames
        # Build index by absolute frame_index (like SpatialClaw)
        self._by_idx: Dict[int, BIMFrame] = {f.frame_index: f for f in frames}

    # --- SpatialClaw-compatible property accessors ----------------------------

    @property
    def depth(self) -> "_IndexAccessor":
        return _IndexAccessor(self._by_idx, lambda f: f.depth)

    @property
    def extrinsics(self) -> "_IndexAccessor":
        return _IndexAccessor(self._by_idx, lambda f: f.camera_to_world)

    @property
    def intrinsics(self) -> "_IndexAccessor":
        return _IndexAccessor(self._by_idx, lambda f: f.intrinsics)

    @property
    def points(self) -> "_IndexAccessor":
        return _IndexAccessor(self._by_idx, lambda f: f.unproject())

    @property
    def frame_indices(self) -> List[int]:
        return sorted(self._by_idx.keys())

    def __len__(self) -> int:
        return len(self.frames)

    def __repr__(self) -> str:
        return (
            f"BIMReconstruction({len(self.frames)} frames, "
            f"frame_indices={self.frame_indices[:4]}{'...' if len(self.frames) > 4 else ''})"
        )

    def render_bev(
        self,
        masks: Optional[List[np.ndarray]] = None,
        labels: Optional[List[str]] = None,
        ref_frame: int = 0,
        resolution: Tuple[int, int] = (1024, 1024),
    ) -> Image.Image:
        """Top-down BEV overlay from reconstructed world-space points.

        Merges all frame depth maps into a single XZ (floor) projection.
        Optionally draws instance masks and labels.

        Returns a PIL.Image in RGB.
        """
        all_xyz: List[np.ndarray] = []
        all_rgb: List[np.ndarray] = []

        for frm in self.frames:
            xyz = frm.unproject()  # H×W×3
            H, W, _ = xyz.shape
            rgb_arr = np.array(frm.rgb.resize((W, H)), dtype=np.float32)
            valid = np.isfinite(xyz[:, :, 0])
            pts = xyz[valid]          # N×3
            clr = rgb_arr[valid]      # N×3
            all_xyz.append(pts)
            all_rgb.append(clr)

        if not all_xyz:
            return Image.new("RGB", resolution, (245, 245, 245))

        pts = np.concatenate(all_xyz, axis=0)
        clr = np.concatenate(all_rgb, axis=0).astype(np.uint8)

        x, y = pts[:, 0], pts[:, 2]  # use X and Z for floor plan (Y is up)

        W_img, H_img = resolution
        margin = 0.05
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = max(x_max - x_min, 1e-3)
        y_range = max(y_max - y_min, 1e-3)

        px = ((x - x_min) / x_range * (1 - 2 * margin) + margin) * W_img
        py = ((y_max - y) / y_range * (1 - 2 * margin) + margin) * H_img
        px = np.clip(px.astype(np.int32), 0, W_img - 1)
        py = np.clip(py.astype(np.int32), 0, H_img - 1)

        canvas = np.full((H_img, W_img, 3), (245, 245, 245), dtype=np.uint8)
        canvas[py, px] = clr

        return Image.fromarray(canvas, mode="RGB")


class _IndexAccessor:
    """Allows ``reconstruction.depth[frame_idx]`` style access."""

    def __init__(self, by_idx: Dict[int, BIMFrame], fn):
        self._by_idx = by_idx
        self._fn = fn

    def __getitem__(self, frame_idx: int):
        if frame_idx not in self._by_idx:
            raise KeyError(
                f"Frame {frame_idx} not in reconstruction. "
                f"Available: {sorted(self._by_idx.keys())}"
            )
        return self._fn(self._by_idx[frame_idx])


class RGBDLoader:
    """Loads BIM-Walker outputs into BIMFrame + BIMReconstruction objects."""

    def load_frames(
        self,
        rgb_dir: Path,
        depth_exr_dir: Path,
        pose_file: Path,
        camera_info_json: Path,
        frame_indices: Optional[List[int]] = None,
    ) -> List[BIMFrame]:
        """Load a subset of frames from BIM-Walker output directories.

        Parameters
        ----------
        frame_indices:
            List of absolute 0-based indices to load. When None, loads all frames.
        """
        rgb_dir = Path(rgb_dir)
        depth_exr_dir = Path(depth_exr_dir)

        cam_info = json.loads(Path(camera_info_json).read_text())
        fx = float(cam_info["fx"])
        fy = float(cam_info["fy"])
        cx = float(cam_info["cx"])
        cy = float(cam_info["cy"])
        W = int(cam_info["width"])
        H = int(cam_info["height"])

        all_rgb = sorted(rgb_dir.glob("*.png"))
        all_exr = sorted(depth_exr_dir.glob("*.exr"))
        all_poses = _parse_poses(pose_file)

        n = min(len(all_rgb), len(all_exr), len(all_poses))
        if n == 0:
            logger.warning("[RGBDLoader] No frames found in %s", rgb_dir)
            return []

        if frame_indices is None:
            frame_indices = list(range(n))
        else:
            frame_indices = [i for i in frame_indices if i < n]

        frames = []
        for idx in frame_indices:
            try:
                rgb = Image.open(all_rgb[idx]).convert("RGB")
                depth = _load_exr_depth(all_exr[idx])
                pose = all_poses[idx]
                frames.append(BIMFrame(
                    frame_index=idx,
                    rgb=rgb,
                    depth=depth,
                    camera_to_world=pose,
                    fx=fx, fy=fy, cx=cx, cy=cy,
                    width=W, height=H,
                ))
            except Exception as exc:
                logger.warning("[RGBDLoader] Skipping frame %d: %s", idx, exc)

        logger.info("[RGBDLoader] Loaded %d/%d frames.", len(frames), len(frame_indices))
        return frames

    def build_reconstruction(self, frames: List[BIMFrame]) -> BIMReconstruction:
        """Wrap a list of BIMFrames into a BIMReconstruction."""
        return BIMReconstruction(frames)

    def load_from_walker_output(
        self,
        walker_output,
        frame_indices: Optional[List[int]] = None,
    ) -> BIMReconstruction:
        """Convenience: load directly from a WalkerOutput dataclass."""
        frames = self.load_frames(
            rgb_dir=walker_output.rgb_dir,
            depth_exr_dir=walker_output.depth_exr_dir,
            pose_file=walker_output.pose_file,
            camera_info_json=walker_output.camera_info_json,
            frame_indices=frame_indices,
        )
        return self.build_reconstruction(frames)
