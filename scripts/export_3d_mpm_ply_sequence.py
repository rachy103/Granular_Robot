from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if ROOT.as_posix() not in sys.path:
    sys.path.insert(0, ROOT.as_posix())
if SRC.as_posix() not in sys.path:
    sys.path.insert(0, SRC.as_posix())

from granular_mpm import SandMPM3D, SandMPM3DConfig
from granular_mpm.export import write_particle_ply
from scripts.run_3d_blade_demo import blade_state


def run(out_dir: Path, frames: int, substeps: int, pscale: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    solver = SandMPM3D(SandMPM3DConfig(dt=8.0e-4, seed=7), device="cuda:0")
    sim_t = 0.0
    print(f"exporting {frames} PLY frames with {solver.n_particles} MPM material points")
    for frame_id in range(frames):
        for _ in range(substeps):
            solver.step(blade_state(sim_t, solver.config.dt), substeps=1)
            sim_t += solver.config.dt
        pos = solver.positions()
        vel = solver.velocities()
        jp = solver.plastic_volume()
        path = out_dir / f"particles_{frame_id:04d}.ply"
        write_particle_ply(path, pos, vel, jp, pscale=pscale)
        if frame_id % 5 == 0:
            print(
                f"frame={frame_id:04d} t={sim_t:.3f} "
                f"z=[{pos[:,2].min():.3f},{pos[:,2].max():.3f}] {path}"
            )
    print(f"ply_dir={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "ply_sequence")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--substeps", type=int, default=34)
    parser.add_argument("--pscale", type=float, default=0.0055)
    args = parser.parse_args()
    run(args.out, args.frames, args.substeps, args.pscale)


if __name__ == "__main__":
    main()
