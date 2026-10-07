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


def main(args):
    server = viser.ViserServer()
    # server.gui.configure_theme(dark_mode=True)

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    URDF_DIR = BASE_DIR / "resources" / "robots"
    urdf_path = URDF_DIR / "quadrotor.urdf"

    batch_size = args.batch_size

    # server.initial_camera.position = (-0.5, 2.0, 1.5)
    server.initial_camera.position = (2.0, 0.0, 1.0)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    server.scene.add_grid(
        "/floor",
        width=6.0,
        height=6.0,
        plane="xy",
        cell_size=0.05,
        section_size=1.0,
    )

    pos = np.zeros((batch_size, 3), dtype=np.float32)
    rot = np.zeros((batch_size, 4), dtype=np.float32)
    rot[:, 0] = 1.0
    scales = np.ones(batch_size, dtype=np.float32)

    bases = []
    robots = []

    for i in range(batch_size):
        # 1. Instantiate the Real Robot
        node_name = "/robot_" + str(i)
        robot_base = server.scene.add_frame(node_name, show_axes=False)
        viser_robot = ViserUrdf(
            server, urdf_or_path=urdf_path, root_node_name=node_name
        )
        robot_base.position = pos[i]
        robot_base.wxyz = rot[i]
        bases.append(robot_base)

        robots.append(viser_robot)

    ##############################
    ## Load and play trajectory ##
    ##############################
    init_file = args.file if args.file is not None else ""
    file_input = server.gui.add_text("Trajectory File", initial_value=init_file)
    button = server.gui.add_button("Play Trajectory")

    @button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        assert client is not None

        file_path = file_input.value
        print(f"Loading trajectory from: {file_path}")

        try:
            data = torch.load(file_path)
            states = data["x"]
            controls = data["u"]
            x_des = data["x_des"]
        except Exception as e:
            print(f"Error loading file '{file_path}': {e}")
            client.add_notification(
                "File Error",
                f'Could not load from "{file_path}".',
                color="red",
            )
            return

        horizon = states.shape[1]

        # # ---------------------------------------------------------
        # # Instantiate Blue Ghost Robots along the Trajectory Path
        # # ---------------------------------------------------------
        # num_trail_ghosts = 10  # Number of ghosts to show the path per batch element
        # trail_indices = np.linspace(0, horizon - 1, num=num_trail_ghosts, dtype=int)
        #
        # for batch_idx in range(args.batch_size):
        #     for idx, t_idx in enumerate(trail_indices):
        #         x_trail = states[batch_idx, t_idx].numpy()
        #
        #         trail_node_name = f"/trail_ghost_{batch_idx}_{idx}"
        #         trail_base = server.scene.add_frame(trail_node_name, show_axes=False)
        #
        #         # Position and orientation from the trajectory at timestep t_idx
        #         trail_base.position = x_trail[0:3]
        #         trail_base.wxyz = x_trail[3:7]
        #
        #         ViserUrdf(
        #             server,
        #             urdf_or_path=urdf_path,
        #             root_node_name=trail_node_name,
        #             mesh_color_override=[0.0, 0.0, 1.0, 0.3],  # Blue with transparency
        #         )
        # # ---------------------------------------------------------

        images = []
        t = 0  # Fixed: Start from 0 instead of 50

        while True:
            x = states[:, t].numpy()
            for i in range(args.batch_size):
                bases[i].position = x[i, 0:3]
                bases[i].wxyz = x[i, 3:7]

            t += 1  # Fixed: Uncommented to allow animation progression
            time.sleep(0.1)

            if args.record:
                images.append(client.get_render(height=1080, width=1920))

            if t >= horizon:  # Safely check >= instead of ==
                t = 0
                time.sleep(1.0)
                # Remove or comment out 'break' if you want it to loop continuously
                break

        if args.record:
            print("Generating and sending GIF...")
            client.send_file_download(
                "image.gif", iio.imwrite("<bytes>", images, extension=".gif", loop=0)
            )
            print("Done!")

    #####################
    ## Snapshot Button ##
    #####################
    snapshot_button = server.gui.add_button("Take Snapshot")

    @snapshot_button.on_click
    def _(event: viser.GuiEvent) -> None:
        client = event.client
        if client is None:
            return

        print("Taking snapshot...")

        # Request a render from the client's current camera view
        image_pixels = client.get_render(height=1080, width=1920)

        # Encode the image as a PNG and send it to the browser for download
        client.send_file_download(
            "snapshot.png", iio.imwrite("<bytes>", image_pixels, extension=".png")
        )

        client.add_notification(
            "Snapshot Taken",
            "Image downloaded as snapshot.png",
            color="green",
        )
        print("Snapshot downloaded!")

    while True:
        time.sleep(0.1)
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-batch_size", type=int, help="Number of robots to visualize", default=4
    )
    parser.add_argument("-file", type=str, help="Trajectory file for playback")
    # Updated to action="store_true" for proper boolean handling in argparse
    parser.add_argument("-record", action="store_true", help="Record playback to GIF")
    main(parser.parse_args())
