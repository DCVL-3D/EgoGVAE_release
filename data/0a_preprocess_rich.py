import dataclasses
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from loguru import logger as guru
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from egoallo_original.src.egoallo.preprocessing.body_model import (
    KEYPT_VERTS,
    SMPL_JOINTS,
    BodyModel,
    reflect_pose_aa,
    reflect_root_trajectory,
    run_smpl,
)
from egoallo_original.src.egoallo.preprocessing.geometry import convert_rotation, joints_global_to_local
from egoallo_original.src.egoallo.preprocessing.util import move_to
from egoallo_original.src.egoallo.preprocessing.geometry.rotation import axis_angle_to_matrix, matrix_to_axis_angle


def determine_floor_height_and_contacts(
    body_joint_seq,
    fps,
    vis=False,
    floor_vel_thresh=0.005,
    floor_height_offset=0.01,
    contact_vel_thresh=0.005,  # 0.015
    contact_toe_height_thresh=0.04,  # if static toe above this height
    contact_ankle_height_thresh=0.08,
    terrain_height_thresh=0.04,
    root_height_thresh=0.04,
    cluster_size_thresh=0.25,
    discard_terrain_seqs=False,  # throw away person steps onto objects (determined by a heuristic)
):
    """
    Taken from
    https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py

    Input: body_joint_seq N x 21 x 3 numpy array
    Contacts are N x 4 where N is number of frames and each row is left heel/toe, right heel/toe
    """
    num_frames = body_joint_seq.shape[0]

    # compute toe velocities
    root_seq = body_joint_seq[:, SMPL_JOINTS["hips"], :]
    left_toe_seq = body_joint_seq[:, SMPL_JOINTS["leftToeBase"], :]
    right_toe_seq = body_joint_seq[:, SMPL_JOINTS["rightToeBase"], :]
    left_toe_vel = np.linalg.norm(left_toe_seq[1:] - left_toe_seq[:-1], axis=1)
    left_toe_vel = np.append(left_toe_vel, left_toe_vel[-1])
    right_toe_vel = np.linalg.norm(right_toe_seq[1:] - right_toe_seq[:-1], axis=1)
    right_toe_vel = np.append(right_toe_vel, right_toe_vel[-1])

    if vis:
        plt.figure()
        steps = np.arange(num_frames)
        plt.plot(steps, left_toe_vel, "-r", label="left vel")
        plt.plot(steps, right_toe_vel, "-b", label="right vel")
        plt.legend()
        plt.show()
        plt.close()

    # now foot heights (z is up)
    left_toe_heights = left_toe_seq[:, 2]
    right_toe_heights = right_toe_seq[:, 2]
    root_heights = root_seq[:, 2]

    if vis:
        plt.figure()
        steps = np.arange(num_frames)
        plt.plot(steps, left_toe_heights, "-r", label="left toe height")
        plt.plot(steps, right_toe_heights, "-b", label="right toe height")
        plt.plot(steps, root_heights, "-g", label="root height")
        plt.legend()
        plt.show()
        plt.close()

    # filter out heights when velocity is greater than some threshold (not in contact)
    all_inds = np.arange(left_toe_heights.shape[0])
    left_static_foot_heights = left_toe_heights[left_toe_vel < floor_vel_thresh]
    left_static_inds = all_inds[left_toe_vel < floor_vel_thresh]
    right_static_foot_heights = right_toe_heights[right_toe_vel < floor_vel_thresh]
    right_static_inds = all_inds[right_toe_vel < floor_vel_thresh]

    all_static_foot_heights = np.append(
        left_static_foot_heights, right_static_foot_heights
    )
    all_static_inds = np.append(left_static_inds, right_static_inds)

    if vis:
        plt.figure()
        steps = np.arange(left_static_foot_heights.shape[0])
        plt.plot(steps, left_static_foot_heights, "-r", label="left static height")
        plt.legend()
        plt.show()
        plt.close()

    discard_seq = False
    if all_static_foot_heights.shape[0] > 0:
        cluster_heights = []
        cluster_root_heights = []
        cluster_sizes = []
        # cluster foot heights and find one with smallest median
        clustering = DBSCAN(eps=0.005, min_samples=3).fit(
            all_static_foot_heights.reshape(-1, 1)
        )
        all_labels = np.unique(clustering.labels_)
        # print(all_labels)
        if vis:
            plt.figure()
        min_median = min_root_median = float("inf")
        for cur_label in all_labels:
            cur_clust = all_static_foot_heights[clustering.labels_ == cur_label]
            cur_clust_inds = np.unique(
                all_static_inds[clustering.labels_ == cur_label]
            )  # inds in the original sequence that correspond to this cluster
            if vis:
                plt.scatter(
                    cur_clust, np.zeros_like(cur_clust), label="foot %d" % (cur_label)
                )
            # get median foot height and use this as height
            cur_median = np.median(cur_clust)
            cluster_heights.append(cur_median)
            cluster_sizes.append(cur_clust.shape[0])

            # get root information
            cur_root_clust = root_heights[cur_clust_inds]
            cur_root_median = np.median(cur_root_clust)
            cluster_root_heights.append(cur_root_median)
            if vis:
                plt.scatter(
                    cur_root_clust,
                    np.zeros_like(cur_root_clust),
                    label="root %d" % (cur_label),
                )

            # update min info
            if cur_median < min_median:
                min_median = cur_median
                min_root_median = cur_root_median

        # print(cluster_heights)
        # print(cluster_root_heights)
        # print(cluster_sizes)
        if vis:
            plt.show()
            plt.close()

        floor_height = min_median
        offset_floor_height = (
            floor_height - floor_height_offset
        )  # toe joint is actually inside foot mesh a bit

        if discard_terrain_seqs:
            # print(min_median + TERRAIN_HEIGHT_THRESH)
            # print(min_root_median + ROOT_HEIGHT_THRESH)
            for cluster_root_height, cluster_height, cluster_size in zip(
                cluster_root_heights, cluster_heights, cluster_sizes
            ):
                root_above_thresh = cluster_root_height > (
                    min_root_median + root_height_thresh
                )
                toe_above_thresh = cluster_height > (min_median + terrain_height_thresh)
                cluster_size_above_thresh = cluster_size > int(
                    cluster_size_thresh * fps
                )
                if root_above_thresh and toe_above_thresh and cluster_size_above_thresh:
                    discard_seq = True
                    print("DISCARDING sequence based on terrain interaction!")
                    break
    else:
        floor_height = offset_floor_height = 0.0

    # now find contacts (feet are below certain velocity and within certain range of floor)
    # compute heel velocities
    left_heel_seq = body_joint_seq[:, SMPL_JOINTS["leftFoot"], :]
    right_heel_seq = body_joint_seq[:, SMPL_JOINTS["rightFoot"], :]
    left_heel_vel = np.linalg.norm(left_heel_seq[1:] - left_heel_seq[:-1], axis=1)
    left_heel_vel = np.append(left_heel_vel, left_heel_vel[-1])
    right_heel_vel = np.linalg.norm(right_heel_seq[1:] - right_heel_seq[:-1], axis=1)
    right_heel_vel = np.append(right_heel_vel, right_heel_vel[-1])

    left_heel_contact = left_heel_vel < contact_vel_thresh
    right_heel_contact = right_heel_vel < contact_vel_thresh
    left_toe_contact = left_toe_vel < contact_vel_thresh
    right_toe_contact = right_toe_vel < contact_vel_thresh

    # compute heel heights
    left_heel_heights = left_heel_seq[:, 2] - floor_height
    right_heel_heights = right_heel_seq[:, 2] - floor_height
    left_toe_heights = left_toe_heights - floor_height
    right_toe_heights = right_toe_heights - floor_height

    left_heel_contact = np.logical_and(
        left_heel_contact, left_heel_heights < contact_ankle_height_thresh
    )
    right_heel_contact = np.logical_and(
        right_heel_contact, right_heel_heights < contact_ankle_height_thresh
    )
    left_toe_contact = np.logical_and(
        left_toe_contact, left_toe_heights < contact_toe_height_thresh
    )
    right_toe_contact = np.logical_and(
        right_toe_contact, right_toe_heights < contact_toe_height_thresh
    )

    contacts = np.zeros((num_frames, len(SMPL_JOINTS)))
    contacts[:, SMPL_JOINTS["leftFoot"]] = left_heel_contact
    contacts[:, SMPL_JOINTS["leftToeBase"]] = left_toe_contact
    contacts[:, SMPL_JOINTS["rightFoot"]] = right_heel_contact
    contacts[:, SMPL_JOINTS["rightToeBase"]] = right_toe_contact

    # hand contacts
    left_hand_contact = detect_joint_contact(
        body_joint_seq,
        "leftHand",
        floor_height,
        contact_vel_thresh,
        contact_ankle_height_thresh,
    )
    right_hand_contact = detect_joint_contact(
        body_joint_seq,
        "rightHand",
        floor_height,
        contact_vel_thresh,
        contact_ankle_height_thresh,
    )
    contacts[:, SMPL_JOINTS["leftHand"]] = left_hand_contact
    contacts[:, SMPL_JOINTS["rightHand"]] = right_hand_contact

    # knee contacts
    left_knee_contact = detect_joint_contact(
        body_joint_seq,
        "leftLeg",
        floor_height,
        contact_vel_thresh,
        contact_ankle_height_thresh,
    )
    right_knee_contact = detect_joint_contact(
        body_joint_seq,
        "rightLeg",
        floor_height,
        contact_vel_thresh,
        contact_ankle_height_thresh,
    )
    contacts[:, SMPL_JOINTS["leftLeg"]] = left_knee_contact
    contacts[:, SMPL_JOINTS["rightLeg"]] = right_knee_contact

    return offset_floor_height, contacts, discard_seq


def detect_joint_contact(
    body_joint_seq, joint_name, floor_height, vel_thresh, height_thresh
):
    """
    Taken from
    https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py
    """
    # calc velocity
    joint_seq = body_joint_seq[:, SMPL_JOINTS[joint_name], :]
    joint_vel = np.linalg.norm(joint_seq[1:] - joint_seq[:-1], axis=1)
    joint_vel = np.append(joint_vel, joint_vel[-1])
    # determine contact by velocity
    joint_contact = joint_vel < vel_thresh
    # compute heights
    joint_heights = joint_seq[:, 2] - floor_height
    # compute contact by vel + height
    joint_contact = np.logical_and(joint_contact, joint_heights < height_thresh)

    return joint_contact


def compute_root_align_mats(root_orient):
    """
    Taken from
    https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py

    compute world to canonical frame for each timestep (rotation around up axis)
    """
    root_orient = torch.as_tensor(root_orient).reshape(-1, 3)
    # convert aa to matrices
    root_orient_mat = convert_rotation(root_orient, "aa", "mat").numpy()

    # rotate root so aligning local body right vector (-x) with world right vector (+x)
    #       with a rotation around the up axis (+z)
    # in body coordinates body x-axis is to the left
    body_right = -root_orient_mat[:, :, 0]
    world2aligned_mat, world2aligned_aa = compute_align_from_body_right(body_right)

    return world2aligned_mat


def compute_joint_align_mats(joint_seq):
    """
    Taken from
    https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py

    Compute world to canonical frame for each timestep (rotation around up axis)
    from the given joint seq (T x J x 3)
    """
    left_idx = SMPL_JOINTS["leftUpLeg"]
    right_idx = SMPL_JOINTS["rightUpLeg"]

    body_right = joint_seq[:, right_idx] - joint_seq[:, left_idx]
    body_right = body_right / np.linalg.norm(body_right, axis=1)[:, np.newaxis]

    world2aligned_mat, world2aligned_aa = compute_align_from_body_right(body_right)

    return world2aligned_mat


def compute_align_from_body_right(body_right):
    """
    Taken from
    https://github.com/davrempe/humor/blob/main/humor/scripts/process_amass_data.py
    """
    world2aligned_angle = np.arccos(
        body_right[:, 0] / (np.linalg.norm(body_right[:, :2], axis=1) + 1e-8)
    )  # project to world x axis, and compute angle
    body_right[:, 2] = 0.0
    world2aligned_axis = np.cross(body_right, np.array([[1.0, 0.0, 0.0]]))

    world2aligned_aa = (
        world2aligned_axis
        / (np.linalg.norm(world2aligned_axis, axis=1)[:, np.newaxis] + 1e-8)
    ) * world2aligned_angle[:, np.newaxis]

    world2aligned_mat = convert_rotation(
        torch.as_tensor(world2aligned_aa).reshape(-1, 3), "aa", "mat"
    ).numpy()

    return world2aligned_mat, world2aligned_aa






def load_seq_smpl_params(input_path: str, num_betas: int = 16):
    guru.info(f"Loading from {input_path}")

    # load in input data
    # we leave out "dmpls" and "marker_data"/"marker_label" which are not present in all datasets
    bdata = np.load(input_path)
    fps = 30
    trans = bdata["transl"]  # global translation
    num_frames = len(trans)
    root_orient = bdata["global_orient"]  # global root orientation (1 joint)
    pose_body = bdata["body_pose"]  # body joint rotations (21 joints)
    
    betas_10 = bdata["betas"][:, :10]
    padding = np.zeros((num_frames, 6))
    betas_16 = np.hstack([betas_10, padding])
    
    #######################################################################################
    # Correct yz
    trans_t = torch.from_numpy(trans)
    root_orient_t = torch.from_numpy(root_orient)

    rot_mat = torch.tensor([
        [1, 0, 0],
        [0, 0, 1],
        [0, -1, 0]
    ], dtype=torch.float32)

    trans_t = torch.matmul(trans_t, rot_mat.T)
    
    root_orient_mat_t = axis_angle_to_matrix(root_orient_t)
    new_root_orient_mat_t = rot_mat @ root_orient_mat_t
    root_orient_t = matrix_to_axis_angle(new_root_orient_mat_t)

    trans = trans_t.numpy()
    root_orient = root_orient_t.numpy()
    #######################################################################################

    
    model_vars = {
        "trans": trans,
        "root_orient": root_orient,
        "pose_body": pose_body,
        "betas": betas_16,
    }
    meta = {"fps": fps, "num_frames": num_frames}
    guru.info(f"meta {meta}")
    guru.info(f"model var shapes {str({k: v.shape for k, v in model_vars.items()})}")
    return model_vars, meta


def run_batch_smpl(
    body_model: BodyModel,
    device: torch.device,
    num_total: int,
    batch_size: int,
    return_verts: bool = True,
    **kwargs,
):
    var_dims = body_model.var_dims
    var_names = [name for name in kwargs if name in var_dims]
    model_vars = {
        name: torch.as_tensor(kwargs[name], dtype=torch.float32).reshape(
            -1, var_dims[name]
        )
        for name in var_names
    }
    fopts = {k: v for k, v in kwargs.items() if k not in var_names}

    batch_joints, batch_verts = [], []
    for sidx in range(0, num_total, batch_size):
        eidx = min(sidx + batch_size, num_total)
        batch_model_vars = move_to(
            {name: x[sidx:eidx].contiguous() for name, x in model_vars.items()}, device
        )
        with torch.no_grad():
            joints, verts, _ = run_smpl(
                body_model, return_verts=return_verts, **batch_model_vars, **fopts
            )
        batch_joints.append(joints.detach().cpu())
        if return_verts and verts is not None:
            batch_verts.append(verts.detach().cpu())

    joints_all = torch.cat(batch_joints, dim=0)
    verts_all = torch.cat(batch_verts, dim=0) if len(batch_verts) > 0 else None
    return joints_all, verts_all


def process_seq(
    input_path: str,
    out_path: str,
    smplh_root: str,
    dev_id: int,
    reflect: bool = False,
    overwrite: bool = False,
    **kwargs,
):
    if not overwrite and os.path.isfile(out_path):
        guru.info(f"{out_path} already exists, skipping.")
        return
    guru.info(f"process {input_path} to {out_path}")

    model_vars, meta = load_seq_smpl_params(input_path)

    process_seq_data(
        model_vars, meta, out_path, dev_id, smplh_root, reflect=reflect, **kwargs
    )


def process_seq_data(
    model_vars: Dict,
    meta: Dict,
    out_path: str,
    dev_id: int,
    smplh_root: str,
    reflect: bool = False,
    split_frame_limit: int = 2000,
    discard_shorter_than: float = 1.0, 
    out_fps: int = 30,
    save_verts: bool = False,
):
    guru.info(f"Processing seq with meta {meta}")
    start_t = time.time()

    src_fps = meta["fps"]
    num_frames = meta["num_frames"]

    # only keep middle 80% of sequences to avoid redundanct static poses
    sidx, eidx = int(0.1 * num_frames), int(0.9 * num_frames)
    num_frames = eidx - sidx
    for name, x in model_vars.items():
        model_vars[name] = x[sidx:eidx]
    guru.info(str({k: v.shape for k, v in model_vars.items()}))

    # discard if shorter than threshold
    if num_frames < discard_shorter_than * src_fps:
        guru.info(f"Sequence shorter than {discard_shorter_than} s, discarding...")
        return

    # must do SMPL forward pass to get joints
    # split into manageable chunks to avoid running out of GPU memory for SMPL
    device = (
        torch.device(f"cuda:{dev_id}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # <HACKS>
    # smplx tries to read shape properties, even when use_pca=False
    from smplx.utils import Struct

    Struct.hands_componentsl = np.zeros(100)  # type: ignore
    Struct.hands_componentsr = np.zeros(100)  # type: ignore
    Struct.hands_meanl = np.zeros(100)  # type: ignore
    Struct.hands_meanr = np.zeros(100)  # type: ignore

    # This defaults to 300, but we have 16 beta parameters. When
    # 16<300 the SMPL class will set num_betas to 10...
    from smplx import SMPLH

    assert SMPLH.SHAPE_SPACE_DIM in (300, 16)
    SMPLH.SHAPE_SPACE_DIM = 16

    body_model = BodyModel(f"{smplh_root}/neutral/model.npz", use_pca=False).to(device)
    model_vars = {k: torch.as_tensor(v).float() for k, v in model_vars.items()}
    if reflect:
        rot_og = model_vars["root_orient"]
        rot_re, model_vars["pose_body"] = reflect_pose_aa(
            rot_og, model_vars["pose_body"]
        )
        out = body_model.forward(betas=model_vars["betas"][:1].to(device))
        root_loc = out.Jtr[:, 0].cpu()  # type: ignore
        model_vars["root_orient"], model_vars["trans"] = reflect_root_trajectory(
            rot_og, model_vars["trans"], rot_re, root_loc
        )

    body_joint_seq, body_vtx_seq = run_batch_smpl(
        body_model,
        device,
        num_frames,
        split_frame_limit,
        return_verts=save_verts,
        **model_vars,
    )
    joints_glob = body_joint_seq[:, : len(SMPL_JOINTS), :]
    joint_seq = joints_glob.numpy()

    guru.info(f"Recovered joints and verts {joint_seq.shape}")

    out_dict = model_vars.copy()
    
    
    out_dict["joints"] = joint_seq
    out_dict["joints_loc"], _ = joints_global_to_local(
        convert_rotation(model_vars["root_orient"], "aa", "mat"),
        model_vars["trans"],
        joints_glob,
    )

    if save_verts and body_vtx_seq is not None:
        out_dict["mojo_verts"] = body_vtx_seq[:, KEYPT_VERTS, :].numpy()

    # determine floor height and foot contacts
    floor_height, contacts, discard_seq = determine_floor_height_and_contacts(
        joint_seq, src_fps
    )

    if discard_seq:
        guru.info("Terrain interaction detected, discarding...")
        return

    guru.info(f"Floor height: {floor_height}")
    # translate so floor is at z=0
    for name in ["trans", "joints", "mojo_verts"]:
        if name not in out_dict:
            continue
        out_dict[name][..., 2] -= floor_height

    # compute rotation to canonical frame (forward facing +y) for every frame
    world2aligned_rot = compute_root_align_mats(model_vars["root_orient"])

    out_dict.update(
        {
            "contacts": contacts,
            "floor_height": floor_height,
            "world2aligned_rot": world2aligned_rot,
        }
    )

    meta = {
        "fps": out_fps,
        "num_frames": num_frames,

    }

    guru.info(f"Seq process time: {time.time() - start_t} s")
    guru.info(f"Saving data to {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **meta, **out_dict)


@dataclasses.dataclass
class Config:
    data_root: str = "/home/dev2/Drive_C/KHW/_egotext/head2motion/dataset/rich/val/combined"
    smplh_root: str = "/home/dev2/Drive_C/KHW/_egotext/head2motion/dataset/smplh"
    out_root: str = "/home/dev2/Drive_C/KHW/_egotext/head2motion/dataset/rich/preprocessed_val"
    devices: tuple[int, ...] = (0,)
    overwrite: bool = False



def main(cfg: Config):
    data_root_path = Path(cfg.data_root)
    out_root_path = Path(cfg.out_root)
    
    paths_to_process = list(data_root_path.glob("*.npz"))
    guru.info(f"Found {len(paths_to_process)} files to process.")
    
    dev_ids = cfg.devices
    guru.info(f"Using devices: {dev_ids}")
    
    (out_root_path / "neutral").mkdir(parents=True, exist_ok=True)

    device = dev_ids[0] if dev_ids else "cpu"
    
    guru.info("processing in sequence")
    for i, path in tqdm(enumerate(paths_to_process)):

        out_path = out_root_path / "neutral" / path.name
        r_out_path = out_root_path / "neutral" / f"{path.stem}_reflect{path.suffix}"

        process_seq(
            path,
            out_path,
            cfg.smplh_root,
            device,
            reflect=False,
            overwrite=cfg.overwrite,
        )
        process_seq(
            path,
            r_out_path,
            cfg.smplh_root,
            device,
            reflect=True,
            overwrite=cfg.overwrite,
        )
        
    return



if __name__ == "__main__":
    main(tyro.cli(Config))
