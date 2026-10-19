"""Visualize merged EgoGVAE outputs with an interactive Viser server.

Expected data format for each sample:
    T_world_cpf: (T, 7) or (T, 4, 4), the camera pose in world coordinates
    body_quats:  (T, 21, 4) or (N, T, 21, 4), in WXYZ order
    betas:       SMPL-H body-shape coefficients

Optional fields:
    contacts:    (T, 21) or (N, T, 21)
    base_root:   (T, 7) or (N, T, 7); overrides the CPF-derived root
"""

import re
import time
import traceback
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from typing import Any, Callable

import imageio.v3 as iio
import joblib
import numpy as np
import torch
import trimesh
import tyro
import viser
import viser.transforms as vtf
from scipy.spatial.transform import Rotation

from utils.from_egoallo import fncsmpl, fncsmpl_extensions, network
from utils.from_egoallo.transforms import SE3, SO3


def natural_key(name: str):
    """Sort filenames with embedded numeric IDs in natural order."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", name)]


def as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def normalize_t_world_cpf(value: Any) -> np.ndarray:
    """Return CPF poses in (T, 7) WXYZ + XYZ format."""
    array = np.array(as_numpy(value), dtype=np.float32, copy=True)
    if array.ndim == 3 and array.shape[0] == 1 and array.shape[-1] == 7:
        array = array[0]

    if array.ndim == 2 and array.shape[-1] == 7:
        norms = np.maximum(np.linalg.norm(array[:, :4], axis=-1, keepdims=True), 1e-8)
        array[:, :4] /= norms
        return array

    if array.ndim == 3 and array.shape[-2:] == (4, 4):
        xyzw = Rotation.from_matrix(array[:, :3, :3]).as_quat()
        wxyz = xyzw[:, [3, 0, 1, 2]]
        return np.concatenate([wxyz, array[:, :3, 3]], axis=-1).astype(np.float32)

    raise ValueError(
        f"Invalid T_world_cpf shape {array.shape}; expected (T, 7), "
        "(1, T, 7), or (T, 4, 4)."
    )


def load_ego_sample(
    data_item: dict,
    body_model: fncsmpl.SmplhModel,
    device: torch.device,
):
    """Compute SMPL-H forward kinematics for the EgoGVAE output format."""
    required_keys = ("T_world_cpf", "body_quats", "betas")
    missing = [key for key in required_keys if key not in data_item]
    if missing:
        raise KeyError(f"Missing EgoGVAE output fields: {', '.join(missing)}")

    cpf_np = normalize_t_world_cpf(data_item["T_world_cpf"])
    world_cpf = torch.as_tensor(cpf_np, dtype=torch.float32, device=device)

    body_quats = torch.as_tensor(data_item["body_quats"], dtype=torch.float32, device=device)
    if body_quats.ndim == 3:
        body_quats = body_quats.unsqueeze(0)
    if body_quats.ndim != 4 or body_quats.shape[-2:] != (21, 4):
        raise ValueError(f"Invalid body_quats shape: {tuple(body_quats.shape)}")

    num_samples, timesteps = body_quats.shape[:2]
    if len(cpf_np) != timesteps:
        raise ValueError(
            f"Frame count mismatch: T_world_cpf has {len(cpf_np)} frames, "
            f"but body_quats has {timesteps}."
        )

    betas = torch.as_tensor(data_item["betas"], dtype=torch.float32, device=device)
    if betas.ndim == 1:
        betas = betas[None, None, :]
    elif betas.ndim == 2:
        if betas.shape[0] == num_samples:
            betas = betas[:, None, :]
        elif num_samples == 1 and betas.shape[0] == timesteps:
            betas = betas[None, :, :]
        else:
            raise ValueError(f"Invalid betas shape: {tuple(betas.shape)}")
    if betas.ndim != 3 or betas.shape[0] != num_samples:
        raise ValueError(f"Invalid betas shape: {tuple(betas.shape)}")
    if betas.shape[1] == 1:
        betas = betas.expand(-1, timesteps, -1)
    elif betas.shape[1] != timesteps:
        raise ValueError(f"Invalid betas timestep count: {betas.shape[1]}")

    if "contacts" in data_item:
        contacts = torch.as_tensor(data_item["contacts"], dtype=torch.float32, device=device)
        if contacts.ndim == 2:
            contacts = contacts[None, ...]
        if contacts.shape != (num_samples, timesteps, 21):
            raise ValueError(f"Invalid contacts shape: {tuple(contacts.shape)}")
    else:
        contacts = torch.zeros((num_samples, timesteps, 21), device=device)

    trajectory = network.EgoDenoiseTraj(
        betas=betas,
        body_rotmats=SO3(body_quats).as_matrix(),
        contacts=contacts,
        hand_rotmats=None,
    )
    body_quats = SO3.from_matrix(trajectory.body_rotmats).wxyz
    shaped = body_model.with_shape(torch.mean(trajectory.betas, dim=1, keepdim=True))

    fk_outputs = shaped.with_pose_decomposed(
        T_world_root=SE3.identity(device=device, dtype=body_quats.dtype).parameters(),
        body_quats=body_quats,
        left_hand_quats=None,
        right_hand_quats=None,
    )

    if "base_root" in data_item:
        world_root = torch.as_tensor(data_item["base_root"], dtype=torch.float32, device=device)
        if world_root.ndim == 2:
            world_root = world_root.unsqueeze(0)
        if world_root.shape != (num_samples, timesteps, 7):
            raise ValueError(f"Invalid base_root shape: {tuple(world_root.shape)}")
    else:
        world_root = fncsmpl_extensions.get_T_world_root_from_cpf_pose(
            fk_outputs, world_cpf[None, ...]
        )

    fk_outputs = fk_outputs.with_new_T_world_root(world_root)
    return cpf_np, shaped, fk_outputs, num_samples, timesteps


def setup_visualization(
    server: viser.ViserServer,
    data_item: dict,
    body_model: fncsmpl.SmplhModel,
    device: torch.device,
    playback_ui: tuple[Any, Any, Any],
    glasses_mesh: trimesh.Trimesh | None,
    render_output_dir: Path,
) -> tuple[Callable[[], None], Any, list[Any]]:
    """Populate the scene and return a per-frame update callback."""
    gui_timestep, gui_playing, gui_fps = playback_ui
    cpf_np, shaped, fk_outputs, num_samples, timesteps = load_ego_sample(
        data_item, body_model, device
    )

    gui_timestep.max = timesteps - 1
    gui_timestep.value = 0
    ui_handles = []

    with server.gui.add_folder("Visual Options") as visual_folder:
        ui_handles.append(visual_folder)
        gui_traj_step = server.gui.add_slider("Traj Step", 1, 50, 1, 10)
        gui_show_traj = server.gui.add_checkbox("Show Trajectory", False)
        gui_show_traj_lines = server.gui.add_checkbox("Connect Trajectory Lines", False)
        gui_show_outline = server.gui.add_checkbox("Show Outline", True)
        gui_outline_thick = server.gui.add_slider(
            "Outline Thickness", 0.0, 0.05, 0.001, 0.002
        )

    with server.gui.add_folder("Video Rendering") as render_folder:
        ui_handles.append(render_folder)
        render_range = server.gui.add_multi_slider(
            "Range", min=0, max=timesteps - 1, step=1,
            initial_value=(0, timesteps - 1),
        )
        render_btn = server.gui.add_button("Render MP4")

    content_root = server.scene.add_frame("/dynamic_content", show_axes=False)
    server.scene.add_frame("/dynamic_content/trajectory", show_axes=False)
    cpf_handle = server.scene.add_frame(
        "/dynamic_content/cpf", show_axes=True, axes_length=0.12, axes_radius=0.01
    )

    if glasses_mesh is not None:
        server.scene.add_mesh_trimesh(
            "/dynamic_content/cpf/glasses", glasses_mesh, scale=0.001 * 1.05
        )

    body_handles = []
    outline_handles = []
    faces = body_model.faces.cpu().numpy()
    skin_weights = body_model.weights.cpu().numpy()
    bone_count = body_model.get_num_joints() + 1
    for i in range(num_samples):
        vertices = shaped.verts_zero[i, 0].detach().cpu().numpy()
        joints = shaped.joints_zero[i, 0].detach().cpu().numpy()
        bone_positions = np.concatenate([np.zeros((1, 3)), joints], axis=0)

        body_handles.append(server.scene.add_mesh_skinned(
            f"/dynamic_content/human/{i}",
            vertices=vertices,
            faces=faces,
            skin_weights=skin_weights,
            bone_wxyzs=vtf.SO3.identity(batch_axes=(bone_count,)).wxyz,
            bone_positions=bone_positions,
            color=(0.6, 0.6, 0.8),
        ))

        normals = trimesh.Trimesh(
            vertices=vertices, faces=faces, process=False
        ).vertex_normals
        outline_handles.append(server.scene.add_mesh_skinned(
            f"/dynamic_content/outline/{i}",
            vertices=vertices + normals * gui_outline_thick.value,
            faces=faces[:, ::-1],
            skin_weights=skin_weights,
            bone_wxyzs=vtf.SO3.identity(batch_axes=(bone_count,)).wxyz,
            bone_positions=bone_positions,
            color=(0.05, 0.05, 0.05),
        ))

    @gui_outline_thick.on_update
    def update_outline(_):
        for i in range(num_samples):
            vertices = shaped.verts_zero[i, 0].detach().cpu().numpy()
            normals = trimesh.Trimesh(
                vertices=vertices, faces=faces, process=False
            ).vertex_normals
            outline_handles[i].vertices = vertices + normals * gui_outline_thick.value

    def update_trajectory(t: int):
        if not gui_show_traj.value or t == 0:
            return
        if t % gui_traj_step.value == 0:
            name = f"/dynamic_content/trajectory/f{t}"
            server.scene.add_frame(
                name, wxyz=cpf_np[t, :4], position=cpf_np[t, 4:],
                show_axes=True, axes_length=0.12, axes_radius=0.01,
            )
            if glasses_mesh is not None:
                server.scene.add_mesh_trimesh(
                    f"{name}/glasses", glasses_mesh, scale=0.001
                )
        if gui_show_traj_lines.value:
            server.scene.add_spline_catmull_rom(
                f"/dynamic_content/trajectory/line_{t}",
                positions=cpf_np[t - 1:t + 1, 4:],
                color=(0.8, 0.2, 0.2), line_width=3.0,
            )

    def update_scene(t: int, render_mode: bool = False):
        cpf_handle.wxyz = cpf_np[t, :4]
        cpf_handle.position = cpf_np[t, 4:]

        for i in range(num_samples):
            outline_handles[i].visible = gui_show_outline.value
            for bone_id, (body_bone, outline_bone) in enumerate(
                zip(body_handles[i].bones, outline_handles[i].bones)
            ):
                pose = (
                    fk_outputs.T_world_root[i, t]
                    if bone_id == 0
                    else fk_outputs.Ts_world_joint[i, t, bone_id - 1]
                ).detach().cpu().numpy()
                body_bone.wxyz = outline_bone.wxyz = pose[:4]
                body_bone.position = outline_bone.position = pose[4:]

        if render_mode:
            update_trajectory(t)
        else:
            # Remove any trajectory traces left by a previous video export.
            server.scene.add_frame("/dynamic_content/trajectory", show_axes=False)

    @render_btn.on_click
    def render_mp4(_):
        if not server.get_clients():
            print("[Error] Open the Viser page in a browser before rendering.")
            return

        gui_playing.value = False
        start, end = map(int, render_range.value)
        frames = []
        server.scene.add_frame("/dynamic_content/trajectory", show_axes=False)

        for t in range(start, end + 1):
            gui_timestep.value = t
            update_scene(t, render_mode=True)
            server.flush()
            time.sleep(0.2)

            try:
                client = next(iter(server.get_clients().values()))
                frames.append(client.camera.get_render(height=1080, width=1920))
                print(f"[Render] Frame {t - start + 1}/{end - start + 1}", end="\r")
            except Exception as error:
                print(f"\n[Error] Failed to render frame {t}: {error}")
                break

        if frames:
            render_output_dir.mkdir(parents=True, exist_ok=True)
            filename = render_output_dir / f"render_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            iio.imwrite(filename, frames, fps=int(gui_fps.value))
            print(f"\n[Render] Saved {filename}")

    prev_time = time.time()

    def loop_callback() -> None:
        nonlocal prev_time
        if gui_playing.value:
            now = time.time()
            if now - prev_time > 1.0 / gui_fps.value:
                gui_timestep.value = (gui_timestep.value + 1) % timesteps
                prev_time = now
        update_scene(int(gui_timestep.value))

    return loop_callback, content_root, ui_handles


def main(
    data_root_dir: Path = Path("./outputs/merged_results"),
    smplh_npz_path: Path = Path("./data/smplh/neural/model.npz"),
    glasses_stl_path: Path = Path("./data/glasses.stl"),
    render_output_dir: Path = Path("./outputs/rendered_videos"),
) -> None:
    """Start a Viser viewer for EgoGVAE reconstruction results."""
    if not smplh_npz_path.is_file():
        raise FileNotFoundError(f"SMPL-H model not found: {smplh_npz_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    body_model = fncsmpl.SmplhModel.load(smplh_npz_path).to(device)
    server = viser.ViserServer()
    server.gui.configure_theme(dark_mode=False)
    server.scene.add_grid(
        "/ground", plane="xy", width=5, height=15,
        cell_color=(0.3, 0.3, 0.3), section_color=(0.5, 0.5, 0.5),
    )

    glasses_mesh = None
    if glasses_stl_path.is_file():
        glasses_mesh = trimesh.load(glasses_stl_path, force="mesh")
        glasses_mesh.visual.face_colors = [20, 20, 20, 255]
    else:
        print(f"[Info] Optional glasses mesh not found: {glasses_stl_path}")

    def list_files() -> list[str]:
        if not data_root_dir.is_dir():
            return ["None"]
        files = [p.relative_to(data_root_dir).as_posix()
                 for p in data_root_dir.rglob("*.p")]
        return ["None"] + sorted(files, key=natural_key)

    file_dropdown = server.gui.add_dropdown("File (.p)", options=list_files())
    refresh_button = server.gui.add_button("Refresh Files")

    @refresh_button.on_click
    def refresh_files(_):
        file_dropdown.options = list_files()

    sample_folder = server.gui.add_folder("Sample Selection")
    with server.gui.add_folder("Playback Controls"):
        gui_timestep = server.gui.add_slider("Timestep", 0, 100, 1, 0)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_fps = server.gui.add_slider("FPS", 1, 60, 1, 30)

    state = {
        "current_file": "None",
        "current_sample": None,
        "data_dict": {},
        "sample_handles": [],
        "render_handles": [],
        "content_root": None,
        "loop_callback": lambda: None,
    }

    def clear_scene():
        state["loop_callback"] = lambda: None
        if state["content_root"] is not None:
            state["content_root"].remove()
            state["content_root"] = None
        for handle in state["render_handles"]:
            handle.remove()
        state["render_handles"] = []

    try:
        while True:
            state["loop_callback"]()
            if state["current_file"] != file_dropdown.value:
                state["current_file"] = file_dropdown.value
                state["current_sample"] = None
                state["data_dict"] = {}
                clear_scene()

                for handle in state["sample_handles"]:
                    handle.remove()
                state["sample_handles"] = []

                if state["current_file"] != "None":
                    try:
                        state["data_dict"] = joblib.load(
                            data_root_dir / state["current_file"]
                        )
                        keys = sorted(state["data_dict"].keys(), key=lambda key: natural_key(str(key)))
                        if not keys:
                            continue

                        with sample_folder:
                            sample_dropdown = server.gui.add_dropdown(
                                "Sample Index", options=[str(key) for key in keys],
                                initial_value=str(keys[0]),
                            )
                            load_button = server.gui.add_button("Visualize Sample")
                        state["sample_handles"] = [sample_dropdown, load_button]

                        def load_selected_sample():
                            selected = sample_dropdown.value
                            sample_key = next(key for key in keys if str(key) == selected)
                            if state["current_sample"] == sample_key:
                                return
                            clear_scene()
                            callback, root, ui_handles = setup_visualization(
                                server, state["data_dict"][sample_key], body_model,
                                device, (gui_timestep, gui_playing, gui_fps),
                                glasses_mesh, render_output_dir,
                            )
                            state["current_sample"] = sample_key
                            state["content_root"] = root
                            state["loop_callback"] = callback
                            state["render_handles"] = ui_handles

                        load_button.on_click(lambda _: load_selected_sample())
                        load_selected_sample()
                    except Exception as error:
                        print(f"[Error] Could not load sample: {error}")
                        traceback.print_exc()
            time.sleep(0.001)
    except KeyboardInterrupt:
        print("\n[Info] Shutting down the Viser server.")


if __name__ == "__main__":
    tyro.cli(main)
