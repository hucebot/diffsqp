import argparse
import json
import time
import imageio.v3 as iio
from pathlib import Path
import torch
import bard

import numpy as np
import viser
from viser.extras import ViserUrdf


def resolve_urdf_path(filename):
    if filename is None:
        return None

    BASE_DIR = Path(__file__).resolve().parent.parent
    URDF_DIR = BASE_DIR / "resources" / "robots"
    urdf_path = URDF_DIR / filename
    print(urdf_path)

    return urdf_path


def compute_grid_transforms(batch_size, spacing_x, spacing_y, n_rows, n_cols):

    if n_rows is None or n_cols is None:
        n_rows = int(np.ceil(np.sqrt(batch_size)))
        n_cols = n_rows

    if batch_size > n_rows * n_cols:
        print(
            f"Error: Batch size ({batch_size}) exceeds the proposed grid size ({n_rows} x {n_cols})."
        )
        n_rows = int(np.ceil(np.sqrt(batch_size)))
        n_cols = n_rows
        print(f"Defaulting to square grid: {n_rows} x {n_cols}.")

    x = np.arange(n_cols) - (n_cols - 1) / 2
    y = np.arange(n_rows) - (n_rows - 1) / 2
    xx, yy = np.meshgrid(x, y)

    grid_size = n_rows * n_cols
    positions = np.zeros((grid_size, 3), dtype=np.float32)
    positions[:, 0] = xx.flatten() * spacing_x
    positions[:, 1] = yy.flatten() * spacing_y
    positions[:, 2] = 0.0

    # Truncate to the exact number of requested instances
    positions = positions[:batch_size]

    # All instances have identity rotation.
    rotations = np.zeros((batch_size, 4), dtype=np.float32)
    rotations[:, 0] = 1.0

    return positions, rotations


def main(args):
    server = viser.ViserServer()
    # server.gui.configure_theme(dark_mode=True)

    print("Press Ctrl+C to exit")

    server.scene.add_grid(
        "/floor",
        width=6.0,
        height=6.0,
        plane="xy",
        cell_size=0.25,
        section_size=1.0,
        position=np.array([0.0, 0.0, -0.5]),
    )

    robot_urdf = resolve_urdf_path(args.urdf)
    print(robot_urdf)

    pos, rot = compute_grid_transforms(
        args.batch_size, args.spacing_x, args.spacing_y, args.n_rows, args.n_cols
    )

    while True:
        time.sleep(0.1)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, help="URDF File")
    # Batch size and grid spacing
    parser.add_argument("--batch_size", type=int, help="Batch Size", default=1)
    parser.add_argument("--spacing_x", type=float, help="X-Axis Spacing", default=0.5)
    parser.add_argument("--spacing_y", type=float, help="Y-Axis Spacing", default=0.5)
    parser.add_argument("--n_rows", type=int, help="Grid Rows")
    parser.add_argument("--n_cols", type=int, help="Grid Columns")
    # Trajectory file loading
    parser.add_argument("-file", type=str, help="Trajectory file for playback")
    # Video recording
    parser.add_argument("-record", action="store_true", help="Record playback to GIF")
    main(parser.parse_args())
