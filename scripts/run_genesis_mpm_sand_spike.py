"""Genesis MPM.Sand smoke demo with a moving rigid intrusion tool.

This is intentionally small: it answers whether Genesis can simulate and render
granular MPM in our WSL setup without requiring the interactive viewer.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/genesis_mpm_sand_spike"))
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--particle-size", type=float, default=0.022)
    parser.add_argument("--grid-density", type=float, default=28)
    parser.add_argument("--sand-friction-angle", type=float, default=35.0)
    parser.add_argument("--sand-vis-mode", choices=["particle", "recon"], default="particle")
    parser.add_argument("--log-every", type=int, default=4)
    return parser.parse_args()


def init_genesis(backend_name: str):
    if backend_name == "gpu":
        wsl_lib = "/usr/lib/wsl/lib"
        current = os.environ.get("LD_LIBRARY_PATH", "")
        if wsl_lib not in current.split(":"):
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = f"{wsl_lib}:{current}" if current else wsl_lib
            if env.get("GENESIS_WSL_LIB_REEXEC") != "1":
                env["GENESIS_WSL_LIB_REEXEC"] = "1"
                os.execvpe(sys.executable, [sys.executable, *sys.argv], env)
            os.environ.update(env)

    import genesis as gs

    backend = gs.cpu if backend_name == "cpu" else gs.gpu
    gs.init(backend=backend, logging_level="warning", seed=7)
    return gs


def build_scene(gs, args: argparse.Namespace):
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2.0e-3, substeps=4),
        coupler_options=gs.options.LegacyCouplerOptions(rigid_mpm=True),
        rigid_options=gs.options.RigidOptions(enable_collision=True),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.70, -0.62, -0.22),
            upper_bound=(0.70, 0.62, 0.82),
            particle_size=args.particle_size,
            grid_density=args.grid_density,
        ),
        vis_options=gs.options.VisOptions(
            visualize_mpm_boundary=False,
            show_world_frame=False,
            ambient_light=(0.42, 0.42, 0.42),
            plane_reflection=False,
        ),
        viewer_options=gs.options.ViewerOptions(
            res=(args.width, args.height),
            camera_pos=(1.10, -1.30, 0.74),
            camera_lookat=(0.02, 0.00, 0.20),
            camera_fov=35,
            max_FPS=int(args.fps),
        ),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )

    scene.add_entity(
        morph=gs.morphs.Plane(),
        surface=gs.surfaces.Default(color=(0.42, 0.43, 0.40, 1.0), roughness=0.95),
    )
    sand = scene.add_entity(
        morph=gs.morphs.Box(pos=(0.02, 0.0, 0.18), size=(0.46, 0.34, 0.20)),
        material=gs.materials.MPM.Sand(
            E=5.0e5,
            nu=0.2,
            rho=1500.0,
            friction_angle=args.sand_friction_angle,
            sampler="random",
        ),
        surface=gs.surfaces.Default(color=(0.78, 0.60, 0.36, 1.0), roughness=1.0, vis_mode=args.sand_vis_mode),
    )
    tool = scene.add_entity(
        morph=gs.morphs.Box(pos=(-0.42, 0.0, 0.24), size=(0.12, 0.43, 0.13), fixed=True),
        material=gs.materials.Rigid(
            friction=0.85,
            needs_coup=True,
            coup_friction=0.80,
            coup_softness=0.0015,
        ),
        surface=gs.surfaces.Default(color=(0.10, 0.28, 0.72, 1.0), roughness=0.55),
    )
    cam = scene.add_camera(
        res=(args.width, args.height),
        pos=(1.10, -1.30, 0.76),
        lookat=(0.02, 0.0, 0.20),
        fov=35,
        GUI=False,
    )
    return scene, sand, tool, cam


def tool_pose(frame: int, frames: int) -> tuple[float, float, float]:
    s = frame / max(frames - 1, 1)
    x = -0.42 + 0.72 * s
    z = 0.245 - 0.035 * np.sin(np.pi * s)
    return float(x), 0.0, float(z)


def render_frame(cam) -> np.ndarray:
    rendered = cam.render(rgb=True)
    rgb = rendered[0] if isinstance(rendered, tuple) else rendered
    return np.ascontiguousarray(rgb[..., ::-1])


def collect_particles(sand) -> dict[str, np.ndarray]:
    state = sand.get_state()
    pos = state.pos.detach().cpu().numpy().reshape(-1, 3)
    vel = state.vel.detach().cpu().numpy().reshape(-1, 3)
    active = state.active.detach().cpu().numpy().reshape(-1).astype(bool)
    return {
        "pos": pos[active],
        "vel": vel[active],
    }


def write_contact_sheet(frames: list[np.ndarray], path: Path) -> None:
    if not frames:
        return
    thumbs = [cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2)) for frame in frames]
    rows = []
    for i in range(0, len(thumbs), 2):
        pair = thumbs[i : i + 2]
        if len(pair) == 1:
            pair.append(np.zeros_like(pair[0]))
        rows.append(np.concatenate(pair, axis=1))
    cv2.imwrite(str(path), np.concatenate(rows, axis=0))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "genesis_mpm_sand_rgb.mp4"
    preview_path = args.output_dir / "genesis_mpm_sand_preview.png"
    sheet_path = args.output_dir / "genesis_mpm_sand_sheet.png"
    npz_path = args.output_dir / "genesis_mpm_particle_log.npz"

    gs = init_genesis(args.backend)
    scene, sand, tool, cam = build_scene(gs, args)
    scene.build()

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {video_path}")

    sheet_frames: list[np.ndarray] = []
    particle_pos: list[np.ndarray] = []
    particle_vel: list[np.ndarray] = []
    logged_frames: list[int] = []

    for frame_idx in range(args.frames):
        tool.set_pos(tool_pose(frame_idx, args.frames))
        scene.step()
        frame = render_frame(cam)
        if frame_idx == 0:
            cv2.imwrite(str(preview_path), frame)
        if frame_idx in {0, args.frames // 3, 2 * args.frames // 3, args.frames - 1}:
            sheet_frames.append(frame.copy())
        writer.write(frame)

        if frame_idx % args.log_every == 0 or frame_idx == args.frames - 1:
            particles = collect_particles(sand)
            particle_pos.append(particles["pos"])
            particle_vel.append(particles["vel"])
            logged_frames.append(frame_idx)

    writer.release()
    write_contact_sheet(sheet_frames, sheet_path)
    np.savez_compressed(
        npz_path,
        frame=np.asarray(logged_frames, dtype=np.int32),
        pos=np.asarray(particle_pos, dtype=object),
        vel=np.asarray(particle_vel, dtype=object),
        particle_size=np.asarray(args.particle_size, dtype=np.float32),
        grid_density=np.asarray(args.grid_density, dtype=np.float32),
        sand_friction_angle=np.asarray(args.sand_friction_angle, dtype=np.float32),
    )
    print(f"Video: {video_path}")
    print(f"Preview: {preview_path}")
    print(f"Sheet: {sheet_path}")
    print(f"Particle log: {npz_path}")


if __name__ == "__main__":
    main()
