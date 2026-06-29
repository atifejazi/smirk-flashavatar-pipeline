import os, sys
import random
import json
from PIL import Image
import torch
import math
import numpy as np
from tqdm import tqdm

from scene.gaussian_model import GaussianModel
from scene.cameras import Camera
from arguments import ModelParams
from utils.general_utils import PILtoTensor
from utils.graphics_utils import focal2fov


def _open_image_flexible(folder, stem):
    """Try .png first, then .jpg/.jpeg. Raises FileNotFoundError if neither exists."""
    for ext in ('.png', '.jpg', '.jpeg'):
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return Image.open(p), p
    raise FileNotFoundError(f"No image found for {stem} in {folder} (tried .png/.jpg/.jpeg)")


class Scene_mica:
    def __init__(self, datadir, mica_datadir, train_type, white_background, device,
                 downscale=1.0, max_frames=None, camera_z=None, force_camera_z=False,
                 frame_delta=1):
        ## train_type: 0 for train, 1 for test, 2 for eval
        ## frame_delta: offset between .frame file index and image file index.
        ##   1 = metrical-tracker default (frame 0 → image 1)
        ##   0 = CREMA-D / 0-indexed datasets (frame N → image N)
        images_folder = os.path.join(datadir, "imgs")
        parsing_folder = os.path.join(datadir, "parsing")
        alpha_folder = os.path.join(datadir, "alpha")
        
        # self.bg_image = torch.zeros((3, 512, 512))
        # if white_background:
        #     self.bg_image[:, :, :] = 1
        # else:
        #     self.bg_image[1, :, :] = 1



        if downscale is None:
            scale = 1.0
        else:
            try:
                scale = float(downscale)
            except (TypeError, ValueError):
                scale = 1.0
        if scale <= 0.0 or scale > 1.0:
            scale = 1.0

        if max_frames is not None and max_frames > 0:
            max_frames = int(max_frames)
        else:
            max_frames = None

        mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint')
        frame_files = [f for f in os.listdir(mica_ckpt_dir) if f.endswith(".frame")]
        self.N_frames = len(frame_files)
        self.cameras = []

        # Adapt train/test split for short clips (e.g. CREMA-D with ~65 frames)
        if self.N_frames >= 600:
            test_num = 500
            eval_num = 50
        else:
            test_num = max(5, self.N_frames // 5)
            eval_num = max(2, self.N_frames // 20)

        max_train_num = 10000
        train_num = min(max_train_num, self.N_frames - test_num)
        ckpt_path = os.path.join(mica_ckpt_dir, '00000.frame')
        payload = torch.load(ckpt_path, weights_only=False)
        flame_params = payload['flame']
        self.shape_param = torch.as_tensor(flame_params['shape']).unsqueeze(0)
        if train_type == 0:
            range_down = 0
            range_up = train_num
        elif train_type == 1:
            range_down = max(0, self.N_frames - test_num)
            range_up = self.N_frames
        elif train_type == 2:
            range_down = max(0, self.N_frames - eval_num)
            range_up = self.N_frames
        elif train_type == 3:
            # All frames — used for render_all mode so the output video covers
            # the whole clip (important for short CREMA-D clips).
            range_down = 0
            range_up = self.N_frames

        if max_frames is not None:
            range_up = min(range_up, range_down + max_frames)

        # Cap range so that the corresponding image file always exists.
        img_files = [f for f in os.listdir(images_folder)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if img_files:
            max_img_idx = max(int(os.path.splitext(f)[0]) for f in img_files)
            max_mica_fid = max(int(os.path.splitext(f)[0]) for f in frame_files)
            max_valid_fid = min(max_mica_fid, max_img_idx - frame_delta)
            range_up = min(range_up, max_valid_fid + 1)
        if range_down >= range_up:
            raise RuntimeError(
                f"No frames in range [{range_down}, {range_up}) for {datadir}: "
                f"check MICA checkpoints vs imgs (frame_delta={frame_delta})."
            )

        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5) # obey mica tracking
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'.frame')
            payload = torch.load(ckpt_path, weights_only=False)
            
            flame_params = payload['flame']
            exp_param = torch.as_tensor(flame_params['exp']).unsqueeze(0) # update: unsqueeezd these
            eyes_pose = torch.as_tensor(flame_params['eyes']).unsqueeze(0)
            eyelids = torch.as_tensor(flame_params['eyelids']).unsqueeze(0)
            jaw_pose = torch.as_tensor(flame_params['jaw']).unsqueeze(0)
            # Global head rotation (axis-angle, 3-d). Falls back to zeros for
            # datasets that don't store it (e.g. original metrical-tracker frames).
            global_key = flame_params.get('global', None)
            global_rot = (torch.as_tensor(global_key).unsqueeze(0)
                          if global_key is not None
                          else torch.zeros(1, 3))

            oepncv = payload['opencv']
            w2cR = oepncv['R'][0]
            w2cT = oepncv['t'][0]
            R = np.transpose(w2cR) # R is stored transposed due to 'glm' in CUDA code
            T = w2cT.flatten() #flattened atif update
            if camera_z is not None:
                try:
                    cam_z = float(camera_z)
                except (TypeError, ValueError):
                    cam_z = None
                if cam_z is not None:
                    if force_camera_z or np.linalg.norm(T) < 1.0e-8:
                        T = T.copy()
                        T[2] = cam_z

            image, image_path = _open_image_flexible(images_folder, image_name_ori)
            orig_w, orig_h = payload['img_size']
            actual_w, actual_h = image.size
            K = payload['opencv']['K'][0].copy()
            if (orig_w, orig_h) != (actual_w, actual_h):
                scale_w = actual_w / max(1, orig_w)
                scale_h = actual_h / max(1, orig_h)
                K[0, 0] *= scale_w
                K[1, 1] *= scale_h
                K[0, 2] *= scale_w
                K[1, 2] *= scale_h
                orig_w, orig_h = actual_w, actual_h

            if scale != 1.0:
                K[0, 0] *= scale
                K[1, 1] *= scale
                K[0, 2] *= scale
                K[1, 2] *= scale
                scaled_w = max(1, int(round(orig_w * scale)))
                scaled_h = max(1, int(round(orig_h * scale)))
                image = image.resize((scaled_w, scaled_h), resample=Image.LANCZOS)
            else:
                scaled_w, scaled_h = orig_w, orig_h
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]

            fl_x = K[0, 0]
            fl_y = K[1, 1]
            FovY = focal2fov(fl_y, scaled_h)
            FovX = focal2fov(fl_x, scaled_w)
            
            # alpha
            alpha, _ = _open_image_flexible(alpha_folder, image_name_ori)
            if scale != 1.0:
                alpha = alpha.resize((scaled_w, scaled_h), resample=Image.BILINEAR)
            alpha = PILtoTensor(alpha)

            bg_image = torch.zeros_like(gt_image)
            if white_background:
                bg_image[:, :, :] = 1
            else:
                bg_image[1, :, :] = 1 # Creates the green screen

            gt_image = gt_image * alpha + bg_image * (1 - alpha)

            # # if add head mask
            #head_mask_path = os.path.join(parsing_folder, image_name_ori+'_neckhead.png')
            # head_mask_path = os.path.join(datadir, 'parsing', 'resnet34', f"{image_name_ori}.png")
            # head_mask = Image.open(head_mask_path)
            # head_mask = PILtoTensor(head_mask)
            ##gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            #gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            #mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png')
            # mouth_mask_path = os.path.join(datadir, 'parsing', 'resnet34', f"{image_name_ori}.png")
            # mouth_mask = Image.open(mouth_mask_path)
            # mouth_mask = PILtoTensor(mouth_mask)

            head_mask = alpha  # Use the RVM alpha map as the head mask

            # Load mouth mask from parsing if available (CREMA-D provides _mouth.png).
            mouth_mask_path = os.path.join(parsing_folder, image_name_ori + '_mouth.png')
            if os.path.exists(mouth_mask_path):
                mouth_img = Image.open(mouth_mask_path).convert('L')
                if scale != 1.0:
                    mouth_img = mouth_img.resize((scaled_w, scaled_h), resample=Image.NEAREST)
                mouth_mask = PILtoTensor(mouth_img)
                mouth_mask = (mouth_mask > 0.5).float()
            else:
                mouth_mask = torch.zeros_like(alpha)

            camera_indiv = Camera(colmap_id=frame_id, R=R, T=T, 
                                FoVx=FovX, FoVy=FovY, 
                                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
                                exp_param=exp_param, eyes_pose=eyes_pose, eyelids=eyelids,
                                jaw_pose=jaw_pose, global_rot=global_rot,
                                image_name=image_name_mica, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)
    
    def getCameras(self):
        return self.cameras





    
