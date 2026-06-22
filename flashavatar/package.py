"""
goal: take smirk's track_params.pt into .frame files that FlashAv loader accepts.
extract jaw/eye rot frmo pose isntead of zeroing it in an attempt for better qualtiy results for personal dataset.

std FLAME 15-d pose layout: [global(3), neck(3), jaw(3), eye_l(3), eye_r(3)].

  python package.py --idname name --pt_file track_params.pt
  python package.py --idname cremaD_ident_name --pt_file smirk_outputs/foldername.pt \
                    --img_size 512 512 --focal 1200
"""

import argparse
import os
import sys

import numpy as np
import torch


def axis_angle_to_matrix_np(aa):
    """convert an axis-angle (3,) numpy array to a 3x3 rotation matrix (Rodrigues)."""
    aa = np.asarray(aa, dtype=np.float32).reshape(3)
    theta = float(np.linalg.norm(aa))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float32)
    k = aa / theta
    K = np.array([[0.0, -k[2], k[1]],
                  [k[2], 0.0, -k[0]],
                  [-k[1], k[0], 0.0]], dtype=np.float32)
    return (np.eye(3, dtype=np.float32)
            + np.sin(theta) * K
            + (1.0 - np.cos(theta)) * (K @ K))


def pad_or_crop(arr, target_length):
    arr = np.asarray(arr).flatten().astype(np.float32)
    if len(arr) >= target_length:
        return arr[:target_length]
    return np.pad(arr, (0, target_length - len(arr)), "constant")


def get_field(data, name, frame_idx):
    """return data[name][frame_idx] as a flat numpy array"""
    if name not in data:
        return None
    val = data[name]
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu().numpy()
    val = np.asarray(val)
    if val.ndim == 1:
        return val.astype(np.float32)
    return val[frame_idx].astype(np.float32)


def slice_pose(pose_vec, raw_pose_dim):
    """
    split a FLAME-style pose vector into (jaw_3, eyes_6) using the original
    raw dim to decide the layout (because pose_vec is already padded to >=15).

    layouts:
      - 15: [global(3), neck(3), jaw(3), eye_l(3), eye_r(3)]   (full FLAME)
      -  6: [global(3), jaw(3)]                                (SMIRK)
      -  3: [jaw(3) only]
    SMIRK does not predict eye gaze, so eyes default to zeros for size 6.
    """
    pose = np.asarray(pose_vec).flatten().astype(np.float32)
    if raw_pose_dim >= 15:
        jaw = pose[6:9]
        eyes = pose[9:15]
    elif raw_pose_dim >= 6:
        jaw = pose[3:6]
        eyes = np.zeros(6, dtype=np.float32)
    elif raw_pose_dim >= 3:
        jaw = pose[:3]
        eyes = np.zeros(6, dtype=np.float32)
    else:
        jaw = np.zeros(3, dtype=np.float32)
        eyes = np.zeros(6, dtype=np.float32)
    return jaw.astype(np.float32), eyes.astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--idname", required=True, help="Identity name (sub-folder under metrical-tracker/output/).")
    p.add_argument("--pt_file", default="track_params.pt", help="Path to SMIRK track_params.pt (default: track_params.pt).")
    p.add_argument("--out_dir", default=None, help="Output dir; default: metrical-tracker/output/<idname>/checkpoint.")
    p.add_argument("--img_size", type=int, nargs=2, default=[512, 512], metavar=("W", "H"), help="Image (W H). Default 512 512.")
    p.add_argument("--focal", type=float, default=1200.0, help="Focal length in pixels. Used only if pt has no K.")
    p.add_argument("--shape_dim", type=int, default=300, help="FLAME shape dim (default 300).")
    p.add_argument("--exp_dim", type=int, default=100, help="FLAME expression dim. Must match FlashAvatar FLAME config (default 100).")
    p.add_argument("--max_frames", type=int, default=None, help="Optional cap on number of frames to write.")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = args.out_dir or os.path.join("metrical-tracker", "output", args.idname, "checkpoint")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isfile(args.pt_file):
        print(f"[package] error: pt file not found: {args.pt_file}", file=sys.stderr)
        sys.exit(1)

    data = torch.load(args.pt_file, map_location="cpu")
    if "exp" not in data or "shape" not in data or "pose" not in data:
        print("[package] error: pt file must contain 'shape', 'exp', 'pose'.", file=sys.stderr)
        print(f"[package] available keys: {list(data.keys())}", file=sys.stderr)
        sys.exit(1)

    num_frames = int(data["exp"].shape[0])
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)

    actual_exp_dim = int(np.asarray(data["exp"]).shape[-1])
    exp_dim = args.exp_dim
    if actual_exp_dim != exp_dim:
        print(f"[package] NOTE: pt file has {actual_exp_dim}-d exp; will pad/crop to {exp_dim}-d (FlashAvatar FLAME space).")

    img_w, img_h = args.img_size
    if "K" in data:
        K_in = np.asarray(data["K"]).astype(np.float32)
        K_matrix = K_in[0] if K_in.ndim == 3 else K_in
    else:
        K_matrix = np.array(
            [[args.focal, 0.0, img_w / 2.0],
             [0.0, args.focal, img_h / 2.0],
             [0.0, 0.0, 1.0]], dtype=np.float32,
        )

    eyelid_key = next((k for k in ("eyelids", "eyelid_params", "eyelid") if k in data), None)
    jaw_key = next((k for k in ("jaw", "jaw_params") if k in data), None)
    eyes_key = next((k for k in ("eyes", "eye_params", "eye_pose", "eyes_pose") if k in data), None)
    has_R = "R" in data
    has_t = "t" in data

    raw_pose_dim = int(np.asarray(data["pose"]).shape[-1])
    if raw_pose_dim == 6:
        layout = "SMIRK 6-d [global(3), jaw(3)]"
    elif raw_pose_dim == 15:
        layout = "full FLAME 15-d [global, neck, jaw, eye_l, eye_r]"
    else:
        layout = f"unknown ({raw_pose_dim}-d)"

    cam_R_source = "from pt" if has_R else (
        "SMIRK pose[:3]" if raw_pose_dim >= 3 else "identity"
    )
    print(
        f"[package] id='{args.idname}' frames={num_frames} img={img_w}x{img_h} "
        f"pose_layout='{layout}' K_from_pt={'K' in data} cam_R={cam_R_source} "
        f"eyelid_key={eyelid_key!r} jaw_key={jaw_key!r} eyes_key={eyes_key!r}"
    )

    eye_motion_seen = False
    jaw_motion_seen = False
    global_rot_angles = []

    for i in range(num_frames):
        raw_shape = get_field(data, "shape", i)
        raw_exp = get_field(data, "exp", i)
        raw_pose = get_field(data, "pose", i)
        if raw_shape is None or raw_exp is None or raw_pose is None:
            print(f"[package] skipping frame {i}: missing shape/exp/pose", file=sys.stderr)
            continue

        shape_data = pad_or_crop(raw_shape, args.shape_dim)
        exp_data = pad_or_crop(raw_exp, exp_dim)
        pose_data = pad_or_crop(raw_pose, max(15, raw_pose.size))

        jaw, eyes = slice_pose(pose_data, raw_pose.size)

        # Prefer explicit per-frame keys when available — they always win over pose slicing.
        if jaw_key is not None:
            jaw = pad_or_crop(get_field(data, jaw_key, i), 3)
        if eyes_key is not None:
            eyes = pad_or_crop(get_field(data, eyes_key, i), 6)

        if not eye_motion_seen and np.any(np.abs(eyes) > 1e-6):
            eye_motion_seen = True
        if not jaw_motion_seen and np.any(np.abs(jaw) > 1e-6):
            jaw_motion_seen = True

        eyelids = (
            pad_or_crop(get_field(data, eyelid_key, i), 2)
            if eyelid_key is not None
            else np.zeros(2, dtype=np.float32)
        )

        neck = pose_data[3:6] if pose_data.size >= 6 else np.zeros(3, dtype=np.float32)

        # Global head rotation (axis-angle, 3-d). Stored separately so FLAME's
        # forward_geo can rotate around the root joint (correct centering), rather
        # than baking it into the camera which would cause a per-frame position drift.
        global_rot = pose_data[:3].astype(np.float32)
        if not has_R:
            global_rot_angles.append(float(np.linalg.norm(global_rot)))

        if has_R:
            R_in = get_field(data, "R", i).reshape(3, 3)
        else:
            R_in = np.eye(3, dtype=np.float32)
        if has_t:
            t_in = get_field(data, "t", i).reshape(3, 1)
        else:
            t_in = np.zeros((3, 1), dtype=np.float32)

        frame_data = {
            "flame": {
                "shape": shape_data,
                "beta": shape_data,
                "exp": exp_data,
                "pose": pose_data[:15],
                "global": global_rot,
                "eyes": eyes,
                "eyelids": eyelids,
                "jaw": jaw,
                "neck": neck.astype(np.float32),
            },
            "img_size": (img_w, img_h),
            "opencv": {
                "K": [K_matrix],
                "R": [R_in.astype(np.float32)],
                "t": [t_in.astype(np.float32)],
            },
        }

        torch.save(frame_data, os.path.join(out_dir, f"{i:05d}.frame"))

    print(f"[package] wrote {num_frames} .frame files to {out_dir}")
    if global_rot_angles:
        ga = np.asarray(global_rot_angles)
        print(
            f"[package] global head rotation magnitude (rad): "
            f"min={ga.min():.3f} max={ga.max():.3f} mean={ga.mean():.3f} "
            f"(camera R will vary per frame)"
        )
    if not jaw_motion_seen:
        print("[package] WARNING: jaw values are all near zero across frames — lips will look static.")
    if not eye_motion_seen:
        print("[package] WARNING: eye values are all near zero across frames — gaze will look fixed (SMIRK doesn't track gaze).")


if __name__ == "__main__":
    main()
