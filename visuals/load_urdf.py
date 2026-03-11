import argparse
import json
import time
import imageio.v3 as iio
from pathlib import Path
import torch

import numpy as np
import viser
from viser.extras import ViserUrdf


def create_grid_transforms(
    num_instances: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create grid positions, rotations, and scales for mesh instances."""
    grid_size = int(np.ceil(np.sqrt(num_instances)))

    # Create grid positions.
    x = np.arange(grid_size) - (grid_size - 1) / 2
    y = np.arange(grid_size) - (grid_size - 1) / 2
    xx, yy = np.meshgrid(x, y)

    positions = np.zeros((grid_size * grid_size, 3), dtype=np.float32)
    positions[:, 0] = 0.4 * xx.flatten()
    positions[:, 1] = 0.3 * yy.flatten()
    # positions[:, 0] = 0.2 * xx.flatten()
    # positions[:, 1] = 0.2 * yy.flatten()
    positions[:, 2] = 0.5
    positions = positions[:num_instances]

    # All instances have identity rotation.
    rotations = np.zeros((num_instances, 4), dtype=np.float32)
    rotations[:, 0] = 1.0  # w component = 1

    # Initial scales.
    scales = np.linalg.norm(positions, axis=-1)
    scales = np.sin(scales * 1.5) * 0.5 + 1.0
    return positions, rotations, scales.astype(np.float32)


def main(args):
    server = viser.ViserServer()
    # server.gui.configure_theme(dark_mode=True)

    RESOURCES = Path(__file__).resolve().parent
    URDF_DIR = RESOURCES / "robots"

    urdf_path = None
    if args.robot == "cartpole":
        urdf_path = URDF_DIR / "cartpole.urdf"
    elif args.robot == "acrobot":
        urdf_path = URDF_DIR / "acrobot.urdf"
    else:
        exit("ERROR: Robot ", args.robot, " unsupported!")

    n_robots = args.n_robots
    states = torch.load(args.file)
    horizon = len(states)

    server.initial_camera.position = (-0.5, 2.0, 1.5)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    # Scene decoration
    # server.scene.world_axes.visible = True
    server.scene.add_grid(
        "/floor",
        width=6.0,
        height=6.0,
        plane="xy",
        cell_size=0.25,
        section_size=1.0,
    )

    pos, rot, scales = create_grid_transforms(n_robots)
    bases = []
    robots = []
    for i in range(n_robots):
        node_name = "/robot_" + str(i)
        robot_base = server.scene.add_frame(node_name, show_axes=False)
        viser_robot = ViserUrdf(
            server, urdf_or_path=urdf_path, root_node_name=node_name
        )
        robot_base.position = pos[i]
        bases.append(robot_base)
        robots.append(viser_robot)

    button = server.gui.add_button("Play and render a video")

    @button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None
        client.scene.reset()
        images = []
        t = 0
        while True:
            x = states[t][:, 0:2].numpy()
            for i in range(args.n_robots):
                robots[i].update_cfg(x[i])
            # Propagate time
            images.append(client.get_render(height=1080, width=1920))
            t += 1
            if t == len(states):
                t = 0
                # time.sleep(1.0)
                images.append(client.get_render(height=1080, width=1920))
                print("Hi")
                break
            # time.sleep(0.01)

        print("Generating and sending GIF...")
        client.send_file_download(
            "image.gif", iio.imwrite("<bytes>", images, extension=".gif", loop=0)
        )
        print("Done!")

    while True:
        time.sleep(0.01)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-robot", type=str, help="Robot name")
    parser.add_argument("-n_robots", type=int, help="Number of robots to visualize")
    parser.add_argument("-file", type=str, help="Trajectory file for playback")
    main(parser.parse_args())
