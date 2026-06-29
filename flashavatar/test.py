import os, sys 
import random
import numpy as np
import torch
import argparse
import cv2
import time
import datetime

from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    # cli
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='id1_25', help='id name')
    parser.add_argument('--logname', type=str, default='log', help='log name')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution (fallback)')
    parser.add_argument('--downscale', type=float, default=0.5, help='Must match the value used at training time.')
    parser.add_argument('--camera_z', type=float, default=1.0, help='Default camera Z when frame extrinsics are zero.')
    parser.add_argument('--force_camera_z', action='store_true', help='Force camera Z override for all frames.')
    parser.add_argument('--frame_delta', type=int, default=1, help='Offset between .frame index and image index. 1=metrical-tracker (default), 0=CREMA-D/0-indexed.')
    parser.add_argument('--render_all', action='store_true', help='Render all frames (not just the held-out test split). Recommended for short clips.')
    parser.add_argument("--checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    batch_size = 1
    set_random_seed(args.seed)

    ## deform model
    DeformModel = Deform_Model(args.device).to(args.device)
    DeformModel.training_setup()
    DeformModel.eval()

    ## dataloader
    data_dir = os.path.join('dataset', args.idname)
    mica_datadir = os.path.join('metrical-tracker/output', args.idname)
    logdir = data_dir+'/'+args.logname
    scene = Scene_mica(
        data_dir,
        mica_datadir,
        train_type=3 if args.render_all else 1,
        white_background=lpt.white_background,
        device=args.device,
        downscale=args.downscale,
        camera_z=args.camera_z,
        force_camera_z=args.force_camera_z,
        frame_delta=args.frame_delta,
    )
    
    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)

    if args.checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vid_save_path = os.path.join(logdir, 'test.avi')
    out = None

    viewpoint = scene.getCameras().copy()
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    for iteration in range(len(viewpoint)):
        viewpoint_cam = viewpoint[iteration]
        frame_id = viewpoint_cam.uid

        # deform gaussians
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['eyes_pose'] = viewpoint_cam.eyes_pose
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose
        codedict['global_rot'] = viewpoint_cam.global_rot
        verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        # Render
        render_pkg = render(viewpoint_cam, gaussians, ppt, background)
        image= render_pkg["render"]
        image = image.clamp(0, 1)

        gt_image = viewpoint_cam.original_image
        gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
        image_np = (image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

        if image_np.shape[:2] != gt_image_np.shape[:2]:
            image_np = cv2.resize(image_np, (gt_image_np.shape[1], gt_image_np.shape[0]), interpolation=cv2.INTER_AREA)

        h, w = gt_image_np.shape[:2]
        if out is None:
            out = cv2.VideoWriter(vid_save_path, fourcc, 25, (w * 2, h), True)

        save_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
        save_image[:, :w, :] = gt_image_np
        save_image[:, w:, :] = image_np # 
        save_image = save_image[:, :, [2, 1, 0]]

        out.write(save_image)
    if out is not None:
        out.release()
    
    
   
        

           
