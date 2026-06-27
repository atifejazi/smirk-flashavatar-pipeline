import os, sys 
import random
import numpy as np
import torch
import torch.nn as nn
import argparse
import cv2
import lpips

from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import huber_loss
from utils.general_utils import normalize_for_percep


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
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument('--downscale', type=float, default=0.5, help='Downscale factor for input images/masks (1.0=full res; higher=sharper, more VRAM).')
    parser.add_argument('--preview_frames', type=int, default=300, help='Number of frames for preview stage.')
    parser.add_argument('--preview_iters', type=int, default=2000, help='Iterations for preview stage.')
    parser.add_argument('--full_iters', type=int, default=20000, help='Total iterations to run (including preview).')
    parser.add_argument('--skip_preview', action='store_true', help='Skip preview stage and train on full set.')
    parser.add_argument('--camera_z', type=float, default=1.0, help='Default camera Z when frame extrinsics are zero.')
    parser.add_argument('--force_camera_z', action='store_true', help='Force camera Z override for all frames.')
    parser.add_argument('--frame_delta', type=int, default=1, help='Offset between .frame index and image index. 1=metrical-tracker (default), 0=CREMA-D/0-indexed.')
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)
    if args.full_iters is not None:
        opt.iterations = args.full_iters

    batch_size = 1
    set_random_seed(args.seed)

    percep_module = lpips.LPIPS(net='vgg').to(args.device)

    ## deform model
    DeformModel = Deform_Model(args.device).to(args.device)
    DeformModel.training_setup()

    ## dataloader
    data_dir = os.path.join('dataset', args.idname)
    mica_datadir = os.path.join('metrical-tracker/output', args.idname)
    log_dir = os.path.join(data_dir, 'log')
    train_dir = os.path.join(log_dir, 'train')
    model_dir = os.path.join(log_dir, 'ckpt')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    
    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)
    if args.start_checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.start_checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    start_iter = first_iter + 1
    total_iters = opt.iterations
    stages = []
    if (not args.skip_preview and args.preview_iters > 0 and args.preview_frames > 0
            and start_iter <= args.preview_iters):
        preview_end = min(args.preview_iters, total_iters)
        stages.append({"name": "preview", "end": preview_end, "max_frames": args.preview_frames})
        if preview_end < total_iters:
            stages.append({"name": "full", "end": total_iters, "max_frames": None})
    else:
        stages.append({"name": "full", "end": total_iters, "max_frames": None})

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)
    
    codedict = {}
    # use a fraction of the run 
    mid_num = max(1, min((total_iters * 35) // 100, total_iters - 2))
    print(f"[train] total_iters={total_iters}, LPIPS active after iter {mid_num}, downscale={args.downscale}")
    current_iter = start_iter
    for stage in stages:
        scene = Scene_mica(
            data_dir,
            mica_datadir,
            train_type=0,
            white_background=lpt.white_background,
            device=args.device,
            downscale=args.downscale,
            max_frames=stage["max_frames"],
            camera_z=args.camera_z,
            force_camera_z=args.force_camera_z,
            frame_delta=args.frame_delta,
        )
        codedict['shape'] = scene.shape_param.to(args.device)
        if current_iter == start_iter:
            DeformModel.example_init(codedict)

        viewpoint_stack = None
        for iteration in range(current_iter, stage["end"] + 1):
            # 500 iters
            if iteration % 500 == 0:
                gaussians.oneupSHdegree()
                torch.cuda.empty_cache()  # free fragmented VRAM before forward pass

            # random camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getCameras().copy()
                random.shuffle(viewpoint_stack)
                if len(viewpoint_stack)>2000:
                    viewpoint_stack = viewpoint_stack[:2000]
            viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))
            frame_id = viewpoint_cam.uid

            # deform gaussians
            codedict['expr'] = viewpoint_cam.exp_param
            codedict['eyes_pose'] = viewpoint_cam.eyes_pose
            codedict['eyelids'] = viewpoint_cam.eyelids
            codedict['jaw_pose'] = viewpoint_cam.jaw_pose
            codedict['global_rot'] = viewpoint_cam.global_rot
            verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)

            if iteration == 1:
                gaussians.create_from_verts(verts_final[0])
                gaussians.training_setup(opt)
            gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

            # render
            render_pkg = render(viewpoint_cam, gaussians, ppt, background)
            image = render_pkg["render"]

            # loss
            gt_image = viewpoint_cam.original_image
            mouth_mask = viewpoint_cam.mouth_mask

            loss_huber = huber_loss(image, gt_image, 0.1) + 40*huber_loss(image*mouth_mask, gt_image*mouth_mask, 0.1)

            loss_G = 0.
            head_mask = viewpoint_cam.head_mask
            image_percep = normalize_for_percep(image*head_mask)
            gt_image_percep = normalize_for_percep(gt_image*head_mask)
            if iteration > mid_num:
                loss_G = torch.mean(percep_module.forward(image_percep, gt_image_percep)) * 0.06

            loss = loss_huber*1 + loss_G*1

            loss.backward()

            with torch.no_grad():
                # optimizer step
                if iteration < opt.iterations :
                    gaussians.optimizer.step()
                    DeformModel.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                    DeformModel.optimizer.zero_grad(set_to_none = True)

                # print loss
                if iteration % 500 == 0:
                    if iteration<=mid_num:
                        print("step: %d, huber: %.5f" %(iteration, loss_huber.item()))
                    else:
                        print("step: %d, huber: %.5f, percep: %.5f" %(iteration, loss_huber.item(), loss_G.item()))

                # visualize results
                if iteration % 500 == 0 or iteration==1:
                    gt_image_np = (gt_image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                    image = image.clamp(0, 1)
                    image_np = (image*255.).permute(1,2,0).detach().cpu().numpy().astype(np.uint8)
                    if image_np.shape[:2] != gt_image_np.shape[:2]:
                        image_np = cv2.resize(image_np, (gt_image_np.shape[1], gt_image_np.shape[0]), interpolation=cv2.INTER_AREA)
                    h, w = gt_image_np.shape[:2]
                    save_image = np.zeros((h, w * 2, 3), dtype=np.uint8)
                    save_image[:, :w, :] = gt_image_np
                    save_image[:, w:, :] = image_np
                    cv2.imwrite(os.path.join(train_dir, f"{iteration}.png"), save_image[:,:,[2,1,0]])

                # save checkpoint
                if iteration % 5000 == 0:
                    print("\n[ITER {}] Saving Checkpoint".format(iteration))
                    torch.save((DeformModel.capture(), gaussians.capture(), iteration), model_dir + "/chkpnt" + str(iteration) + ".pth")

        current_iter = stage["end"] + 1

           
