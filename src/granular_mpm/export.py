from __future__ import annotations

from pathlib import Path

import numpy as np


def sand_colors(pos: np.ndarray, jp: np.ndarray | None = None) -> np.ndarray:
    z = pos[:, 2]
    u = (z - float(z.min())) / max(1.0e-6, float(z.max() - z.min()))
    base = np.stack(
        [
            145.0 + 68.0 * u,
            103.0 + 58.0 * u,
            47.0 + 36.0 * u,
        ],
        axis=1,
    )
    if jp is not None:
        compaction = np.clip(1.0 - jp, 0.0, 0.35) / 0.35
        base[:, 0] -= 30.0 * compaction
        base[:, 1] -= 22.0 * compaction
        base[:, 2] -= 8.0 * compaction
    return np.clip(base, 0.0, 255.0).astype(np.uint8)


def write_particle_ply(
    path: Path,
    pos: np.ndarray,
    vel: np.ndarray | None = None,
    jp: np.ndarray | None = None,
    pscale: float = 0.0055,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(pos.shape[0])
    if vel is None:
        vel = np.zeros_like(pos)
    if jp is None:
        jp = np.ones(n, dtype=np.float32)
    colors = sand_colors(pos, jp)

    header = "\n".join(
        [
            "ply",
            "format ascii 1.0",
            "comment MPM material points exported for Houdini/Blender/CloudCompare visualization",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
            "property float vx",
            "property float vy",
            "property float vz",
            "property float Jp",
            "property float pscale",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
        ]
    )

    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(header)
        f.write("\n")
        for i in range(n):
            f.write(
                f"{pos[i,0]:.7f} {pos[i,1]:.7f} {pos[i,2]:.7f} "
                f"{vel[i,0]:.7f} {vel[i,1]:.7f} {vel[i,2]:.7f} "
                f"{jp[i]:.7f} {pscale:.7f} "
                f"{int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}\n"
            )
