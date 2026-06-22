"""
Fit proper 100-d FLAME expression parameters from per-frame SMIRK mesh vertices.

Why this is needed:
    SMIRK encodes expression into its own 50-d latent space (not FLAME's expression basis).
    FlashAvatar's deform model feeds expression coefficients into FLAME's blendshape matrix,
    so wrong-basis latents produce garbage geometry. This script solves the inverse problem:
    given SMIRK's posed vertices (correct geometry) and FLAME's blendshape basis, find the
    100-d FLAME expression vector that best reproduces those vertices per frame.

Math:
    FLAME forward: V_posed = LBS( v_template + B_shape@shape + B_exp@exp + pose_blend, T )
    Linearising around exp=0 in posed space:
        V_posed ≈ V_flame0  +  T[:, :3, :3]  @  B_exp  @  exp
    where:
        V_flame0  = FLAME(shape, exp=0, jaw, global)  (zero-expression reference)
        T         = per-vertex LBS blend matrix  (V, 4, 4)
        B_exp     = flame.shapedirs[:, :, 300:400]  (V, 3, 100)
    So per frame: A_frame (V*3, 100) @ exp = (V_smirk - V_flame0).reshape(-1)

Usage (must run from FlashAvatar-code/ root):
    python tools/fit_flame_exp.py
    python tools/fit_flame_exp.py --verts_file dataset/Atif/atif_verts.npy \
                                  --pt_file track_params.pt \
                                  --out_pt track_params_fitted.pt \
                                  --batch 50
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Must be run from FlashAvatar-code/ root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flame import FLAME_mica, parse_args as flame_parse_args
from flame.lbs import batch_rigid_transform, blend_shapes, vertices2joints
from pytorch3d.transforms import (
    matrix_to_rotation_6d, rotation_6d_to_matrix, axis_angle_to_matrix
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--verts_file", default="dataset/Atif/atif_verts.npy",
                   help="Path to (N, 5023, 3) SMIRK vertex array.")
    p.add_argument("--pt_file", default="track_params.pt",
                   help="SMIRK track_params.pt with shape (N,300) and pose (N,6).")
    p.add_argument("--out_pt", default="track_params_fitted.pt",
                   help="Output .pt file with fitted FLAME exp params added.")
    p.add_argument("--batch", type=int, default=50,
                   help="Frames per GPU batch for lstsq (default 50).")
    p.add_argument("--exp_reg", type=float, default=0.01,
                   help="L2 regularisation on expression (Tikhonov lambda, default 0.01).")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def aa_to_rot6d(aa_vec, device):
    """Axis-angle (3,) → rotation-6d (1, 6)."""
    mat = axis_angle_to_matrix(aa_vec.unsqueeze(0).to(device))  # (1, 3, 3)
    return matrix_to_rotation_6d(mat)                           # (1, 6)


def identity_rot6d(device):
    return matrix_to_rotation_6d(torch.eye(3, device=device).unsqueeze(0))  # (1, 6)


def compute_T_matrix(flame, shape_1, global_6d, jaw_6d, device):
    """
    Run FLAME LBS internals for a single frame and return the per-vertex
    blend transform T of shape (V, 4, 4).

    Parameters
    ----------
    flame   : FLAME_mica in eval mode
    shape_1 : (1, 300) shape tensor on device
    global_6d : (1, 6) global rotation in rot6d
    jaw_6d    : (1, 6) jaw rotation in rot6d
    """
    I = identity_rot6d(device)
    full_pose = torch.cat([global_6d, I, jaw_6d, I, I], dim=1)  # (1, 30)

    betas_0 = torch.cat([shape_1, torch.zeros(1, 100, device=device)], dim=1)

    v_template = flame.v_template.unsqueeze(0)                       # (1, V, 3)
    v_shaped = v_template + blend_shapes(betas_0, flame.shapedirs)   # (1, V, 3)

    J = vertices2joints(flame.J_regressor, v_shaped)                  # (1, J, 3)

    rot_mats = rotation_6d_to_matrix(
        full_pose.view(-1, 6)
    ).view(1, -1, 3, 3)  # (1, J, 3, 3)

    ident = torch.eye(3, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).view(1, -1)
    pose_offsets = torch.matmul(pose_feature, flame.posedirs).view(1, -1, 3)
    v_posed = v_shaped + pose_offsets  # (1, V, 3)

    _, A = batch_rigid_transform(rot_mats, J, flame.parents, dtype=torch.float32)
    # A : (1, J, 4, 4)

    W = flame.lbs_weights.unsqueeze(0)              # (1, V, J)
    num_joints = flame.J_regressor.shape[0]
    T = torch.matmul(W, A.view(1, num_joints, 16)).view(1, -1, 4, 4)  # (1, V, 4, 4)
    return T[0], v_posed[0]  # (V, 4, 4), (V, 3)


def fit_batch(A_frames, b_frames, exp_reg):
    """
    Solve a batch of least-squares problems with Tikhonov regularisation.

    A_frames : (B, M, 100)   where M = V*3
    b_frames : (B, M)
    Returns  : (B, 100)
    """
    B, M, K = A_frames.shape
    # Normal equations with regularisation: (A^T A + lambda I) exp = A^T b
    AtA = torch.bmm(A_frames.transpose(1, 2), A_frames)  # (B, 100, 100)
    Atb = torch.bmm(A_frames.transpose(1, 2), b_frames.unsqueeze(-1)).squeeze(-1)  # (B, 100)
    lam = exp_reg * torch.eye(K, device=A_frames.device).unsqueeze(0)
    return torch.linalg.solve(AtA + lam, Atb)  # (B, 100)


def main():
    args = parse_args()
    device = torch.device(args.device)

    print(f"[fit_flame_exp] device={device}")

    # ---- load FLAME model ------------------------------------------------
    cfg = flame_parse_args()
    flame = FLAME_mica(cfg).to(device)
    flame.eval()

    B_exp = flame.shapedirs[:, :, 300:400]   # (V, 3, 100)
    V = B_exp.shape[0]

    # ---- load data --------------------------------------------------------
    print(f"[fit_flame_exp] loading {args.verts_file}")
    verts_np = np.load(args.verts_file)          # (N, 5023, 3)
    assert verts_np.shape[1] == V, f"Expected {V} vertices, got {verts_np.shape[1]}"
    N = verts_np.shape[0]

    print(f"[fit_flame_exp] loading {args.pt_file}")
    data = torch.load(args.pt_file, map_location="cpu")
    shapes = data["shape"]   # (N, 300)
    poses  = data["pose"]    # (N, 6)  SMIRK: [global(3), jaw(3)]
    assert shapes.shape[0] == N, "shape/verts frame count mismatch"

    # ---- pre-compute A per frame and batch solve -------------------------
    fitted_exp = torch.zeros(N, 100, dtype=torch.float32)

    A_buf = []
    b_buf = []
    idx_buf = []

    def flush_batch():
        if not A_buf:
            return
        A_t = torch.stack(A_buf, dim=0).to(device)  # (B, M, 100)
        b_t = torch.stack(b_buf, dim=0).to(device)  # (B, M)
        exp_t = fit_batch(A_t, b_t, args.exp_reg)   # (B, 100)
        for j, fi in enumerate(idx_buf):
            fitted_exp[fi] = exp_t[j].cpu()
        A_buf.clear(); b_buf.clear(); idx_buf.clear()

    print(f"[fit_flame_exp] fitting {N} frames (batch={args.batch}, reg={args.exp_reg})")

    for i in tqdm(range(N)):
        shape_i = shapes[i:i+1].to(device)
        pose_i  = poses[i]

        global_aa = pose_i[:3]
        jaw_aa    = pose_i[3:6]

        global_6d = aa_to_rot6d(global_aa, device)
        jaw_6d    = aa_to_rot6d(jaw_aa,    device)
        I_6d      = identity_rot6d(device)

        with torch.no_grad():
            T_i, _ = compute_T_matrix(flame, shape_i, global_6d, jaw_6d, device)
            # T_i : (V, 4, 4)

            # Zero-expression FLAME reference
            v_flame0_i = flame.forward_geo(
                shape_i,
                rot_params=global_6d,
                jaw_pose_params=jaw_6d,
                expression_params=torch.zeros(1, 100, device=device),
            )  # (1, V, 3)
            v_flame0_i = v_flame0_i[0]  # (V, 3)

        # SMIRK posed verts for this frame
        v_smirk_i = torch.from_numpy(verts_np[i]).to(device)  # (V, 3)

        # Residual in posed space
        residual = (v_smirk_i - v_flame0_i)  # (V, 3)

        # Build system matrix A: T_rot (V, 3, 3) @ B_exp (V, 3, 100) → (V, 3, 100)
        T_rot = T_i[:, :3, :3]          # (V, 3, 3)
        # A_frame[k*3:(k+1)*3, :] = T_rot[k] @ B_exp[k]
        A_frame = torch.einsum("vij,vjk->vik", T_rot, B_exp).reshape(V * 3, 100)  # (M, 100)
        b_frame = residual.reshape(V * 3)                                           # (M,)

        A_buf.append(A_frame.cpu())
        b_buf.append(b_frame.cpu())
        idx_buf.append(i)

        if len(A_buf) >= args.batch:
            flush_batch()

    flush_batch()

    # ---- stats ------------------------------------------------------------
    exp_std = fitted_exp.std(dim=0)
    print(f"[fit_flame_exp] fitted exp per-dim std (first 20): "
          f"{exp_std[:20].tolist()}")
    jaw_motion = (poses[:, 3:6].norm(dim=1) > 0.01).sum().item()
    print(f"[fit_flame_exp] frames with jaw motion > 0.01 rad: {jaw_motion}/{N}")

    # ---- verify reconstruction (sample 5 frames) --------------------------
    print("[fit_flame_exp] reconstruction check (5 random frames):")
    sample_idx = np.random.choice(N, 5, replace=False)
    for si in sorted(sample_idx):
        shape_i = shapes[si:si+1].to(device)
        jaw_6d  = aa_to_rot6d(poses[si, 3:6], device)
        glob_6d = aa_to_rot6d(poses[si, :3],  device)
        exp_i   = fitted_exp[si:si+1].to(device)
        with torch.no_grad():
            v_recon = flame.forward_geo(shape_i, rot_params=glob_6d,
                                        jaw_pose_params=jaw_6d,
                                        expression_params=exp_i)  # (1, V, 3)
        v_recon = v_recon[0].cpu().numpy()
        v_smirk = verts_np[si]
        err_mm = np.linalg.norm(v_recon - v_smirk, axis=-1).mean() * 1000
        print(f"  frame {si:04d}: mean vertex error = {err_mm:.2f} mm")

    # ---- save -------------------------------------------------------------
    out_data = dict(data)
    out_data["exp"] = fitted_exp  # replace SMIRK latent with fitted FLAME exp (N, 100)
    torch.save(out_data, args.out_pt)
    print(f"[fit_flame_exp] saved → {args.out_pt}")
    print("  keys:", list(out_data.keys()))
    print("  exp shape:", out_data["exp"].shape)


if __name__ == "__main__":
    main()
