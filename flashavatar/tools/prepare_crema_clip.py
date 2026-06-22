"""
note:
clips from CREMA_D have this layout:
    <clip_dir>/
        frame_0001.npy, frame_0002.npy, ...   # per frame SMIRK outputs (beta/psi/theta)
        <clip_name>_mesh_sequence.npy          # (N, 5023, 3) posed vertices
        imgs/   00000.jpg ...
        alpha/  00000.jpg ...
        parsing/ 00000_mouth.png, 00000_neckhead.png ...

what happens:
  1. read .npy file per frame
  2. fit proper FLAME expresision coeffs
  3. write .frame files to metrical-tracker/output/<idname>/checkpoint/

then train and test with frame_delta=0:
    tools/run_identity.sh <idname> 0.65 25000 both 0

run using this command from flashav root:
    python tools/prepare_crema_clip.py \\
        --clip_dir  dataset/1001/1001_DFA_ANG_XX \\
        --idname    1001_DFA_ANG_XX
        
and then batch clips

"""

import argparse
import os
import sys
import glob
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flame import FLAME_mica, parse_args as flame_parse_args
from flame.lbs import batch_rigid_transform, blend_shapes, vertices2joints
from pytorch3d.transforms import (
    matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix,
)


# helpers copied from fit_flame_exp.py 

def aa_to_rot6d(aa_vec, device):
    mat = axis_angle_to_matrix(aa_vec.unsqueeze(0).to(device))
    return matrix_to_rotation_6d(mat)


def identity_rot6d(device):
    return matrix_to_rotation_6d(torch.eye(3, device=device).unsqueeze(0))


def compute_T_matrix(flame, shape_1, global_6d, jaw_6d, device):
    I = identity_rot6d(device)
    full_pose = torch.cat([global_6d, I, jaw_6d, I, I], dim=1)  # (1, 30)

    betas_0 = torch.cat([shape_1, torch.zeros(1, 100, device=device)], dim=1)
    v_template = flame.v_template.unsqueeze(0)
    v_shaped = v_template + blend_shapes(betas_0, flame.shapedirs)
    J = vertices2joints(flame.J_regressor, v_shaped)
    rot_mats = rotation_6d_to_matrix(full_pose.view(-1, 6)).view(1, -1, 3, 3)
    ident = torch.eye(3, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).view(1, -1)
    pose_offsets = torch.matmul(pose_feature, flame.posedirs).view(1, -1, 3)
    v_posed = v_shaped + pose_offsets
    _, A = batch_rigid_transform(rot_mats, J, flame.parents, dtype=torch.float32)
    W = flame.lbs_weights.unsqueeze(0)
    num_joints = flame.J_regressor.shape[0]
    T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)
    return T[0], v_posed[0]


def fit_batch(A_frames, b_frames, exp_reg):
    B, M, K = A_frames.shape
    AtA = torch.bmm(A_frames.transpose(1, 2), A_frames)
    Atb = torch.bmm(A_frames.transpose(1, 2), b_frames.unsqueeze(-1)).squeeze(-1)
    lam = exp_reg * torch.eye(K, device=A_frames.device).unsqueeze(0)
    return torch.linalg.solve(AtA + lam, Atb)


def fit_flame_exp(flame, verts_np, shapes_t, poses_t, device,
                  batch_size=50, exp_reg=0.01):
    """
    Fit 100-d FLAME expression coefficients from posed mesh vertices.

    Parameters
    ----------
    flame   : FLAME_mica (eval mode, on device)
    verts_np: (N, V, 3) numpy array of SMIRK-posed vertices
    shapes_t: (N, 300) torch tensor of per-frame FLAME shape params
    poses_t : (N, 6) torch tensor — layout [global_aa(3), jaw_aa(3)]
    device  : torch device

    Returns
    -------
    fitted_exp : (N, 100) torch tensor of fitted FLAME expression params
    """
    B_exp = flame.shapedirs[:, :, 300:400]  # (V, 3, 100)
    V = B_exp.shape[0]
    N = verts_np.shape[0]

    assert verts_np.shape[1] == V, f"Expected {V} vertices, got {verts_np.shape[1]}"

    fitted_exp = torch.zeros(N, 100, dtype=torch.float32)
    A_buf, b_buf, idx_buf = [], [], []

    def flush():
        if not A_buf:
            return
        A_t = torch.stack(A_buf).to(device)
        b_t = torch.stack(b_buf).to(device)
        exp_t = fit_batch(A_t, b_t, exp_reg)
        for j, fi in enumerate(idx_buf):
            fitted_exp[fi] = exp_t[j].cpu()
        A_buf.clear(); b_buf.clear(); idx_buf.clear()

    for i in tqdm(range(N), desc="  fitting FLAME exp"):
        shape_i = shapes_t[i:i+1].to(device)
        pose_i = poses_t[i]
        global_aa = pose_i[:3]
        jaw_aa    = pose_i[3:6]
        global_6d = aa_to_rot6d(global_aa, device)
        jaw_6d    = aa_to_rot6d(jaw_aa, device)

        with torch.no_grad():
            T_i, _ = compute_T_matrix(flame, shape_i, global_6d, jaw_6d, device)
            v_flame0 = flame.forward_geo(
                shape_i,
                rot_params=global_6d,
                jaw_pose_params=jaw_6d,
                expression_params=torch.zeros(1, 100, device=device),
            )[0]

        v_smirk = torch.from_numpy(verts_np[i]).to(device)
        residual = v_smirk - v_flame0
        T_rot = T_i[:, :3, :3]
        A_frame = torch.einsum("vij,vjk->vik", T_rot, B_exp).reshape(V * 3, 100)
        b_frame = residual.reshape(V * 3)

        A_buf.append(A_frame.cpu())
        b_buf.append(b_frame.cpu())
        idx_buf.append(i)
        if len(A_buf) >= batch_size:
            flush()

    flush()
    return fitted_exp


# .frame file writer same format as package.py

def write_frames(out_dir, shapes_np, exp_np, poses_np, img_w, img_h, focal):
    """
    Write per-frame .frame files.

    shapes_np : (N, 300)
    exp_np    : (N, 100)
    poses_np  : (N, 6)  layout [global_aa(3), jaw_aa(3)]
    """
    os.makedirs(out_dir, exist_ok=True)
    N = shapes_np.shape[0]

    K_matrix = np.array(
        [[focal, 0.0,   img_w / 2.0],
         [0.0,   focal, img_h / 2.0],
         [0.0,   0.0,   1.0]], dtype=np.float32,
    )
    R_id = np.eye(3, dtype=np.float32)
    t_zero = np.zeros((3, 1), dtype=np.float32)

    jaw_motion_seen = False

    for i in range(N):
        global_rot = poses_np[i, :3].astype(np.float32)
        jaw        = poses_np[i, 3:6].astype(np.float32)
        shape      = shapes_np[i].astype(np.float32)
        exp        = exp_np[i].astype(np.float32)

        pose15 = np.concatenate([global_rot, np.zeros(3, np.float32), jaw,
                                 np.zeros(6, np.float32)])

        if not jaw_motion_seen and np.any(np.abs(jaw) > 1e-6):
            jaw_motion_seen = True

        frame_data = {
            "flame": {
                "shape":   shape,
                "beta":    shape,
                "exp":     exp,
                "pose":    pose15,
                "global":  global_rot,
                "eyes":    np.zeros(6, np.float32),
                "eyelids": np.zeros(2, np.float32),
                "jaw":     jaw,
                "neck":    np.zeros(3, np.float32),
            },
            "img_size": (img_w, img_h),
            "opencv": {
                "K": [K_matrix],
                "R": [R_id],
                "t": [t_zero],
            },
        }
        torch.save(frame_data, os.path.join(out_dir, f"{i:05d}.frame"))

    return N, jaw_motion_seen



def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--clip_dir", required=True,
                   help="Path to the CREMA-D clip folder "
                        "(e.g. dataset/1001/1001_DFA_ANG_XX).")
    p.add_argument("--idname", required=True,
                   help="FlashAvatar identity name (e.g. 1001_DFA_ANG_XX). "
                        "Will create dataset/<idname> symlink and "
                        "metrical-tracker/output/<idname>/checkpoint/ .frame files.")
    p.add_argument("--focal", type=float, default=1200.0,
                   help="Focal length in pixels (default 1200).")
    p.add_argument("--img_size", type=int, nargs=2, default=[512, 512],
                   metavar=("W", "H"), help="Image size (default 512 512).")
    p.add_argument("--exp_reg", type=float, default=0.01,
                   help="L2 regularisation for expression fitting (default 0.01).")
    p.add_argument("--batch", type=int, default=50,
                   help="Frames per GPU batch during fitting (default 50).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--skip_fit", action="store_true",
                   help="Skip expression fitting and use zero exp (for debugging).")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    clip_dir = os.path.abspath(args.clip_dir)

    if not os.path.isdir(clip_dir):
        print(f"[prepare] ERROR: clip_dir not found: {clip_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"[prepare] clip_dir : {clip_dir}")
    print(f"[prepare] idname   : {args.idname}")
    print(f"[prepare] device   : {device}")

    # per frame .npy
    frame_files = sorted(glob.glob(os.path.join(clip_dir, "frame_*.npy")))
    if not frame_files:
        print("[prepare] ERROR: no frame_XXXX.npy files found in clip_dir.", file=sys.stderr)
        sys.exit(1)

    N = len(frame_files)
    print(f"[prepare] found {N} per-frame .npy files")

    shapes_list, poses_list = [], []
    for fp in frame_files:
        d = np.load(fp, allow_pickle=True).item()
        # beta: FLAME shape (1, 300)
        # theta: global rotation axis-angle (1, 3) 
        beta  = np.asarray(d["beta"]).flatten().astype(np.float32)[:300]
        theta = np.asarray(d["theta"]).flatten().astype(np.float32)[:3]
        # jaw defaults to zeros 
        pose  = np.concatenate([theta, np.zeros(3, np.float32)])
        shapes_list.append(beta)
        poses_list.append(pose)

    shapes_np = np.stack(shapes_list)   # (N, 300)
    poses_np  = np.stack(poses_list)    # (N, 6)

    global_mag = np.linalg.norm(poses_np[:, :3], axis=1)
    print(f"[prepare] global head rotation (rad): "
          f"min={global_mag.min():.3f} max={global_mag.max():.3f} mean={global_mag.mean():.3f}")

    shapes_t = torch.from_numpy(shapes_np)
    poses_t  = torch.from_numpy(poses_np)

    # fit
    mesh_files = sorted(glob.glob(os.path.join(clip_dir, "*_mesh_sequence.npy")))
    if not mesh_files:
        print("[prepare] WARNING: no *_mesh_sequence.npy found; using zero expression.",
              file=sys.stderr)
        fitted_exp_np = np.zeros((N, 100), dtype=np.float32)
    elif args.skip_fit:
        print("[prepare] --skip_fit: using zero expression.")
        fitted_exp_np = np.zeros((N, 100), dtype=np.float32)
    else:
        mesh_file = mesh_files[0]
        if len(mesh_files) > 1:
            print(f"[prepare] multiple mesh files found; using {mesh_file}")
        print(f"[prepare] loading mesh sequence: {mesh_file}")
        verts_np = np.load(mesh_file)   # (N, 5023, 3)
        if verts_np.shape[0] != N:
            # trim
            N_use = min(N, verts_np.shape[0])
            print(f"[prepare] frame count mismatch: {N} .npy files vs "
                  f"{verts_np.shape[0]} mesh frames; using first {N_use}.")
            verts_np   = verts_np[:N_use]
            shapes_np  = shapes_np[:N_use]
            poses_np   = poses_np[:N_use]
            shapes_t   = shapes_t[:N_use]
            poses_t    = poses_t[:N_use]
            N = N_use

        print(f"[prepare] fitting FLAME expression for {N} frames …")
        cfg   = flame_parse_args()
        flame = FLAME_mica(cfg).to(device)
        flame.eval()

        fitted_exp_t = fit_flame_exp(
            flame, verts_np, shapes_t, poses_t, device,
            batch_size=args.batch, exp_reg=args.exp_reg,
        )

        B_exp = flame.shapedirs[:, :, 300:400]
        errors = []
        check_idx = np.linspace(0, N - 1, 3, dtype=int)
        for si in check_idx:
            shape_i  = shapes_t[si:si+1].to(device)
            global_6d = aa_to_rot6d(poses_t[si, :3], device)
            jaw_6d    = aa_to_rot6d(poses_t[si, 3:6], device)
            exp_i     = fitted_exp_t[si:si+1].to(device)
            with torch.no_grad():
                v_recon = flame.forward_geo(shape_i, rot_params=global_6d,
                                            jaw_pose_params=jaw_6d,
                                            expression_params=exp_i)[0]
            err_mm = np.linalg.norm(
                v_recon.cpu().numpy() - verts_np[si], axis=-1).mean() * 1000
            errors.append(err_mm)
            print(f"  frame {si:04d}: mean vertex error = {err_mm:.2f} mm")

        print(f"[prepare] mean recon error across sample frames: "
              f"{np.mean(errors):.2f} mm")
        fitted_exp_np = fitted_exp_t.numpy()

    # .frame files
    out_dir = os.path.join("metrical-tracker", "output", args.idname, "checkpoint")
    img_w, img_h = args.img_size
    n_written, jaw_motion = write_frames(
        out_dir, shapes_np, fitted_exp_np, poses_np, img_w, img_h, args.focal,
    )
    print(f"[prepare] wrote {n_written} .frame files → {out_dir}")
    if not jaw_motion:
        print("[prepare] NOTE: jaw values are all near zero (jaw motion captured "
              "via expression blendshapes from mesh fitting, not axis-angle).")

    link_path = os.path.join("dataset", args.idname)
    if os.path.islink(link_path):
        existing = os.readlink(link_path)
        if os.path.abspath(existing) == clip_dir:
            print(f"[prepare] symlink already correct: {link_path} → {clip_dir}")
        else:
            os.remove(link_path)
            os.symlink(clip_dir, link_path)
            print(f"[prepare] updated symlink: {link_path} → {clip_dir}")
    elif os.path.exists(link_path):
        print(f"[prepare] WARNING: {link_path} already exists as a real directory; "
              "skipping symlink creation. Make sure it contains the clip data.")
    else:
        os.symlink(clip_dir, link_path)
        print(f"[prepare] created symlink: {link_path} → {clip_dir}")

    print()
    print("[prepare] Done! Run with:")
    print(f"  tools/run_identity.sh {args.idname} 0.65 25000 both 0")


if __name__ == "__main__":
    main()
