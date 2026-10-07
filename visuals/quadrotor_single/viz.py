import argparse
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

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    URDF_DIR = BASE_DIR / "resources" / "robots"
    urdf_path = URDF_DIR / "quadrotor.urdf"

    server.initial_camera.position = (2.0, 0.0, 1.0)
    server.initial_camera.look_at = (0.0, 0.0, 0.0)

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    offset = np.array([0.0, 0.0, 0.5])
    server.scene.add_grid(
        "/floor",
        width=6.0,
        height=6.0,
        plane="xy",
        cell_size=0.05,
        section_size=1.0,
    )

    # 1. Instantiate the Real Robot
    real_node_name = "/real_robot"
    real_base = server.scene.add_frame(real_node_name, show_axes=False)
    real_robot = ViserUrdf(
        server, urdf_or_path=urdf_path, root_node_name=real_node_name
    )

    # 2. Instantiate the 4 Noise Ghost Robots
    num_ghosts = 4
    ghost_bases = []
    ghost_robots = []

    for i in range(num_ghosts):
        ghost_node_name = f"/ghost_{i}"
        ghost_base = server.scene.add_frame(ghost_node_name, show_axes=False)
        ghost_robot = ViserUrdf(
            server,
            urdf_or_path=urdf_path,
            root_node_name=ghost_node_name,
            mesh_color_override=[0.0, 0.0, 0.0, 0.2],
        )
        ghost_bases.append(ghost_base)
        ghost_robots.append(ghost_robot)

    start_node_name = f"/start"
    start_base = server.scene.add_frame(start_node_name, show_axes=False)
    start_robot = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name=start_node_name,
        mesh_color_override=[1.0, 0.0, 0.0, 0.5],
    )
    start_base_pos = [0.0, 0.0, 0.0]
    start_base.position = np.array(start_base_pos) + offset

    target_node_name = f"/target"
    target_base = server.scene.add_frame(target_node_name, show_axes=False)
    target_robot = ViserUrdf(
        server,
        urdf_or_path=urdf_path,
        root_node_name=target_node_name,
        mesh_color_override=[0.0, 1.0, 0.0, 0.5],
    )
    target_base_pos = [1.0, 0.0, 0.5]
    target_base.position = np.array(target_base_pos) + offset

    noise_std = torch.tensor([0.01, 0.01, 0.01, 0.001, 0.001, 0.001])
    noise_dim = len(noise_std)

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
            states = data["x"][:, :300, :]
        except Exception as e:
            print(f"Error loading file '{file_path}': {e}")
            client.add_notification(
                "File Error",
                f'Could not load from "{file_path}".',
                color="red",
            )
            return

        horizon = states.shape[1] if states.ndim == 3 else states.shape[0]

        # ---------------------------------------------------------
        # Instantiate Blue Ghost Robots along the Trajectory Path
        # ---------------------------------------------------------
        num_trail_ghosts = 10  # Number of ghosts to show the path
        trail_indices = np.linspace(0, horizon - 1, num=num_trail_ghosts, dtype=int)

        for idx, t_idx in enumerate(trail_indices):
            x_trail = states[0, t_idx] if states.ndim == 3 else states[t_idx]

            trail_node_name = f"/trail_ghost_{idx}"
            trail_base = server.scene.add_frame(trail_node_name, show_axes=False)

            # Position and orientation from the trajectory at timestep t_idx
            trail_base.position = x_trail[0:3].numpy() + offset
            trail_base.wxyz = x_trail[3:7].numpy()

            ViserUrdf(
                server,
                urdf_or_path=urdf_path,
                root_node_name=trail_node_name,
                mesh_color_override=[0.0, 0.0, 1.0, 0.3],  # Blue with transparency
            )
        # ---------------------------------------------------------

        images = []
        t = 0

        while True:
            # Get the true state for the current timestep
            x_true = states[0, t] if states.ndim == 3 else states[t]

            # Update real robot (convert to numpy for viser)
            real_base.position = x_true[0:3].numpy() + offset
            real_base.wxyz = x_true[3:7].numpy()

            # Update the noise ghost bases with the noise logic
            for i in range(num_ghosts):
                noisy_state = x_true.clone()

                # 1. Add noise to the true state
                true_noise = torch.randn((num_ghosts, noise_dim)) * noise_std
                noisy_state[:noise_dim] += true_noise[i]

                # 2. Post-normalize the quaternion of the true state (indices 3 to 6)
                true_quat = noisy_state[3:7]
                noisy_state[3:7] = true_quat / torch.norm(
                    true_quat, p=2, dim=0, keepdim=True
                )

                # Send to viser
                ghost_bases[i].position = noisy_state[0:3].numpy() + offset
                ghost_bases[i].wxyz = noisy_state[3:7].numpy()

            time.sleep(0.1)

            if args.record:
                images.append(client.get_render(height=1080, width=1920))

            t += 1
            if t >= horizon:
                t = 0
                time.sleep(1.0)
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
        image_pixels = client.get_render(height=1080, width=1920)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", type=str, help="Trajectory file for playback")
    parser.add_argument("-record", action="store_true", help="Record playback to GIF")
    main(parser.parse_args())
