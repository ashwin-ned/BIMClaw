"""PointCloudRenderer: PLY point cloud → BEV floor plan image.

Uses pure numpy + matplotlib only — no Open3D required.
BIM-Walker exports a colored PLY (ASCII or binary); we load it, project
top-down (XY plane), and render a height-coloured floor plan.
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def _load_ply(ply_path: Path) -> np.ndarray:
    """Load a PLY file (ASCII or binary_little_endian) and return (N, 6) float32.

    Columns: x, y, z, r, g, b  (rgb in [0, 255]).
    Falls back gracefully when color channels are absent (sets rgb to 128).
    """
    path = Path(ply_path)
    with open(path, "rb") as f:
        raw = f.read()

    lines = raw.split(b"\n")
    header_end = 0
    format_type = "ascii"
    n_vertices = 0
    has_rgb = False

    for i, line in enumerate(lines):
        decoded = line.decode("utf-8", errors="ignore").strip()
        if decoded.startswith("format"):
            format_type = decoded.split()[1]
        if decoded.startswith("element vertex"):
            n_vertices = int(decoded.split()[-1])
        if decoded in ("property uchar red", "property uint8 red"):
            has_rgb = True
        if decoded == "end_header":
            header_end = i + 1
            break

    if n_vertices == 0:
        logger.warning("[PointCloud] PLY has no vertices: %s", path)
        return np.zeros((0, 6), dtype=np.float32)

    if format_type == "ascii":
        data_lines = lines[header_end: header_end + n_vertices]
        rows = []
        for ln in data_lines:
            parts = ln.decode("utf-8", errors="ignore").split()
            if len(parts) >= 3:
                xyz = [float(p) for p in parts[:3]]
                if has_rgb and len(parts) >= 6:
                    rgb = [float(parts[3]), float(parts[4]), float(parts[5])]
                else:
                    rgb = [128.0, 128.0, 128.0]
                rows.append(xyz + rgb)
        return np.array(rows, dtype=np.float32) if rows else np.zeros((0, 6), dtype=np.float32)

    # binary_little_endian — parse header to get property layout
    # Re-read to find byte offset of end_header
    header_bytes = b"\n".join(lines[:header_end]) + b"\n"
    body = raw[len(header_bytes):]

    # BIM-Walker writes: float x, float y, float z, uchar r, uchar g, uchar b
    # = 3×4 + 3×1 = 15 bytes per vertex
    stride = 15
    if len(body) < stride * n_vertices:
        # Try float+float+float only (12 bytes) — old format without color
        stride = 12
        pts = np.frombuffer(body[: stride * n_vertices], dtype=np.float32).reshape(-1, 3)
        gray = np.full((len(pts), 3), 128.0, dtype=np.float32)
        return np.concatenate([pts, gray], axis=1)

    pts = np.frombuffer(body[: stride * n_vertices], dtype=np.uint8).copy()
    # Interpret first 12 bytes as float32, last 3 as uint8 per vertex
    result = np.zeros((n_vertices, 6), dtype=np.float32)
    for i in range(n_vertices):
        off = i * 15
        x, y, z = struct.unpack_from("<fff", body, off)
        r, g, b = body[off + 12], body[off + 13], body[off + 14]
        result[i] = (x, y, z, r, g, b)
    return result


class PointCloudRenderer:
    """Renders BIM-Walker PLY point clouds as 2D images (no GPU, no Open3D)."""

    def render_bev(
        self,
        ply_path: Path,
        out_path: Optional[Path] = None,
        storey_z_range: Optional[Tuple[float, float]] = None,
        resolution: Tuple[int, int] = (1024, 1024),
        colormap: str = "height",
        point_size: int = 2,
        background: Tuple[int, int, int] = (245, 245, 245),
    ) -> Image.Image:
        """Top-down BEV projection of a PLY file.

        Parameters
        ----------
        ply_path:
            Path to the coloured PLY file from BIM-Walker.
        out_path:
            If given, saves the image to this path as PNG.
        storey_z_range:
            (z_min, z_max) in metres to restrict to one storey. When None,
            uses all points.
        resolution:
            Output image size (width, height).
        colormap:
            "height" — colour by Z (rainbow, lower=blue higher=red).
            "rgb"    — use the RGB colors from the PLY file.
        point_size:
            Radius in pixels for each projected point (1–4 recommended).
        background:
            RGB background colour (uint8 tuple).

        Returns
        -------
        PIL.Image (RGB)
        """
        pts = _load_ply(Path(ply_path))
        if len(pts) == 0:
            logger.warning("[PointCloud] Empty PLY, returning blank image.")
            img = Image.new("RGB", resolution, color=background)
            if out_path:
                img.save(out_path)
            return img

        if storey_z_range is not None:
            z_min, z_max = storey_z_range
            mask = (pts[:, 2] >= z_min) & (pts[:, 2] <= z_max)
            pts = pts[mask]
            if len(pts) == 0:
                logger.warning("[PointCloud] No points in Z range [%.1f, %.1f].", z_min, z_max)
                img = Image.new("RGB", resolution, color=background)
                if out_path:
                    img.save(out_path)
                return img

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        r_raw, g_raw, b_raw = pts[:, 3], pts[:, 4], pts[:, 5]

        # Map world XY → pixel XY (flip Y so +Y is up)
        W, H = resolution
        margin = 0.05
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_range = max(x_max - x_min, 1e-3)
        y_range = max(y_max - y_min, 1e-3)

        px = ((x - x_min) / x_range * (1 - 2 * margin) + margin) * W
        py = ((y_max - y) / y_range * (1 - 2 * margin) + margin) * H
        px = np.clip(px.astype(np.int32), 0, W - 1)
        py = np.clip(py.astype(np.int32), 0, H - 1)

        # Colour computation
        if colormap == "rgb":
            colors = np.stack([r_raw, g_raw, b_raw], axis=1).astype(np.uint8)
        else:
            # Height-based rainbow: blue → cyan → green → yellow → red
            t = np.clip((z - z.min()) / max(z.max() - z.min(), 1e-3), 0, 1)
            r_col = np.clip(255 * (1.5 - abs(t - 1.0) * 3), 0, 255)
            g_col = np.clip(255 * (1.5 - abs(t - 0.5) * 3), 0, 255)
            b_col = np.clip(255 * (1.5 - abs(t - 0.0) * 3), 0, 255)
            colors = np.stack([r_col, g_col, b_col], axis=1).astype(np.uint8)

        # Sort by Z so higher points draw on top (painter's algorithm)
        order = np.argsort(z)
        px, py, colors = px[order], py[order], colors[order]

        canvas = np.full((H, W, 3), background, dtype=np.uint8)

        r = max(point_size - 1, 0)
        if r == 0:
            canvas[py, px] = colors
        else:
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        ny = np.clip(py + dy, 0, H - 1)
                        nx = np.clip(px + dx, 0, W - 1)
                        canvas[ny, nx] = colors

        img = Image.fromarray(canvas, mode="RGB")
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path))
            logger.info("[PointCloud] BEV saved to %s", out_path)
        return img

    def render_perspective(
        self,
        ply_path: Path,
        camera_pos: Tuple[float, float, float],
        camera_target: Tuple[float, float, float],
        resolution: Tuple[int, int] = (1024, 768),
        fov_deg: float = 70.0,
        out_path: Optional[Path] = None,
    ) -> Image.Image:
        """Simple pinhole projection of the PLY from an arbitrary viewpoint.

        Uses numpy-only: no GPU.  Good for generating additional viewpoint images.

        Parameters
        ----------
        camera_pos:
            Camera world position (x, y, z).
        camera_target:
            World point the camera looks at.
        fov_deg:
            Horizontal field of view in degrees.
        """
        pts = _load_ply(Path(ply_path))
        if len(pts) == 0:
            return Image.new("RGB", resolution, (245, 245, 245))

        xyz = pts[:, :3]
        r_raw = pts[:, 3].astype(np.uint8)
        g_raw = pts[:, 4].astype(np.uint8)
        b_raw = pts[:, 5].astype(np.uint8)

        cam_pos = np.array(camera_pos, dtype=np.float64)
        cam_target = np.array(camera_target, dtype=np.float64)
        world_up = np.array([0.0, 0.0, 1.0])

        forward = cam_target - cam_pos
        forward /= np.linalg.norm(forward) + 1e-9
        right = np.cross(forward, world_up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])
            right = np.cross(forward, world_up)
            right_norm = np.linalg.norm(right)
        right /= right_norm
        up = np.cross(right, forward)

        # Transform points to camera space
        delta = xyz - cam_pos
        cam_x = delta @ right
        cam_y = delta @ up
        cam_z = delta @ forward

        # Only keep points in front of camera
        mask = cam_z > 0.05
        cam_x, cam_y, cam_z = cam_x[mask], cam_y[mask], cam_z[mask]
        r_f, g_f, b_f = r_raw[mask], g_raw[mask], b_raw[mask]

        if len(cam_z) == 0:
            return Image.new("RGB", resolution, (245, 245, 245))

        W, H = resolution
        half_fov = np.radians(fov_deg / 2)
        f = (W / 2) / np.tan(half_fov)

        px = (cam_x / cam_z * f + W / 2).astype(np.int32)
        py = (-cam_y / cam_z * f + H / 2).astype(np.int32)

        valid = (px >= 0) & (px < W) & (py >= 0) & (py < H)
        px, py = px[valid], py[valid]
        cam_z_v = cam_z[valid]
        r_v, g_v, b_v = r_f[valid], g_f[valid], b_f[valid]

        # Sort back-to-front
        order = np.argsort(-cam_z_v)
        canvas = np.full((H, W, 3), (245, 245, 245), dtype=np.uint8)
        canvas[py[order], px[order]] = np.stack([r_v[order], g_v[order], b_v[order]], axis=1)

        img = Image.fromarray(canvas, mode="RGB")
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            img.save(str(out_path))
        return img


# ---------------------------------------------------------------------------
# CLI entry point for standalone testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render a PLY point cloud as a BEV floor plan")
    parser.add_argument("--ply", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--colormap", choices=["height", "rgb"], default="height")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    renderer = PointCloudRenderer()
    img = renderer.render_bev(
        ply_path=Path(args.ply),
        out_path=Path(args.out),
        resolution=(args.resolution, args.resolution),
        colormap=args.colormap,
    )
    print(f"BEV image saved: {args.out}  ({img.width}×{img.height})")
