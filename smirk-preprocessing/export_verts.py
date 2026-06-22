"""
goal: the per-frame FLAME mesh vertices of the "name" dataset as a single numpy array of shape (N_frames, 5023, 3) for use in a pipeline.

decodes already-saved per-frame SMIRK track_params through the FLAME mesh model so the result is the full mesh (head pose + jaw + expression + eyelids)

command:
    python export_verts.py
"""

import os
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm

from src.FLAME.FLAME import FLAME


def load_track_params(track_dir):
    """dict info by llm:

    Each file is a python dict with keys:
        'shape'  : (300,) float32
        'exp'    : (50,)  float32
        'pose'   : (6,)   float32   -- [global_head_rot(3), jaw_rot(3)]
        'eyelid' : (2,)   float32   (optional; defaults to zeros if missing)

    function returns a dict of stacked tensors of shape (N, ...).
    """
    files = sorted(glob.glob(os.path.join(track_dir, "*.npy")))
    if len(files) == 0:
        raise RuntimeError(f"No .npy files found in {track_dir}")

    shape_list, exp_list, pose_list, eyelid_list = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True).item()
        shape_list.append(np.asarray(d["shape"], dtype=np.float32).reshape(-1))
        exp_list.append(np.asarray(d["exp"], dtype=np.float32).reshape(-1))
        pose_list.append(np.asarray(d["pose"], dtype=np.float32).reshape(-1))
        if "eyelid" in d and d["eyelid"] is not None:
            eyelid_list.append(np.asarray(d["eyelid"], dtype=np.float32).reshape(-1))
        else:
            eyelid_list.append(np.zeros(2, dtype=np.float32))

    shape = np.stack(shape_list, axis=0)      # (N, 300)
    exp = np.stack(exp_list, axis=0)          # (N, 50)
    pose = np.stack(pose_list, axis=0)        # (N, 6)
    eyelid = np.stack(eyelid_list, axis=0)    # (N, 2)

    return {
        "files": files,
        "shape": shape,
        "exp": exp,
        "pose": pose,
        "eyelid": eyelid,
    }


@torch.no_grad()
def decode_vertices(flame, params, device, batch_size=64):
    """Decode all frames into FLAME mesh vertices (N, 5023, 3)."""
    n = params["shape"].shape[0]
    all_verts = np.empty((n, 5023, 3), dtype=np.float32)

    shape_t = torch.from_numpy(params["shape"]).to(device)        # (N, 300)
    exp_t = torch.from_numpy(params["exp"]).to(device)            # (N, 50)
    pose_t = torch.from_numpy(params["pose"]).to(device)          # (N, 6)
    eyelid_t = torch.from_numpy(params["eyelid"]).to(device)      # (N, 2)

    head_pose_t = pose_t[:, :3].contiguous()  # (N, 3) global head rotation
    jaw_pose_t = pose_t[:, 3:].contiguous()   # (N, 3) jaw rotation

    for start in tqdm(range(0, n, batch_size), desc="FLAME decode"):
        end = min(start + batch_size, n)
        param_dict = {
            "shape_params": shape_t[start:end],
            "expression_params": exp_t[start:end],
            "pose_params": head_pose_t[start:end],
            "jaw_params": jaw_pose_t[start:end],
            "eyelid_params": eyelid_t[start:end],
        }
        flame_out = flame.forward(param_dict)
        verts = flame_out["vertices"].detach().cpu().numpy().astype(np.float32)
        if verts.shape[1] != 5023 or verts.shape[2] != 3:
            raise RuntimeError(
                f"Unexpected FLAME vertex shape: {verts.shape} (expected (B, 5023, 3))"
            )
        all_verts[start:end] = verts

    return all_verts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--track_dir",
        type=str,
        default="datasets/atif/track_params",
        help="Directory of per-frame SMIRK track_params .npy files",
    )
    parser.add_argument(
        "--out_verts",
        type=str,
        default="atif_verts.npy",
        help="Output path for the (N, 5023, 3) vertex array",
    )
    parser.add_argument(
        "--out_cam",
        type=str,
        default="atif_cam.npy",
        help="Output path for the per-frame camera dict (or None)",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[info] device = {device}")

    print(f"[info] loading track_params from {args.track_dir}")
    params = load_track_params(args.track_dir)
    n = params["shape"].shape[0]
    print(
        f"[info] loaded {n} frames | shape={params['shape'].shape} "
        f"exp={params['exp'].shape} pose={params['pose'].shape} "
        f"eyelid={params['eyelid'].shape}"
    )

    print("[info] building FLAME model")
    flame = FLAME().to(device).eval()

    print("[info] decoding vertices through FLAME")
    verts = decode_vertices(flame, params, device, batch_size=args.batch_size)
    print(f"[info] verts shape = {verts.shape}, dtype = {verts.dtype}")

    out_verts_abs = os.path.abspath(args.out_verts)
    np.save(out_verts_abs, verts)
    print(f"[ok] saved {out_verts_abs} ({verts.nbytes / 1e6:.2f} MB)")

    # SMIRK's saved track_params do NOT contain camera extrinsics (K/R/t).
    # SMIRK uses an orthographic-style 3-d 'cam' parameter (s, tx, ty) that is
    # produced only at encoder time (not stored in the per-frame track_params).
    # Per the request, we save None when K/R/t aren't available.
    cam_payload = {
        "K": None,
        "R": None,
        "t": None,
        "note": (
            "SMIRK does not produce 3x3 intrinsics or 3x4 extrinsics. "
            "It outputs a 3-d orthographic 'cam' (scale, tx, ty) only at "
            "encoder time, which was NOT saved in the per-frame track_params. "
            "Re-run the SMIRK encoder on imgs/ if you need it."
        ),
    }
    out_cam_abs = os.path.abspath(args.out_cam)
    np.save(out_cam_abs, cam_payload, allow_pickle=True)
    print(f"[ok] saved {out_cam_abs} (camera = None; see 'note' inside)")

    print()
    print("=" * 60)
    print("DONE.")
    print(f"Vertices file: {out_verts_abs}")
    print(f"Frames       : {verts.shape[0]}")
    print(f"Vertices/frame: {verts.shape[1]} (expected 5023)")
    print("=" * 60)


if __name__ == "__main__":
    main()
