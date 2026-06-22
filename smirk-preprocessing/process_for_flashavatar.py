import torch
import cv2
import numpy as np
import os
from tqdm import tqdm
from src.smirk_encoder import SmirkEncoder

# set paths HERE
VIDEO_PATH = 'rvm_output.mp4' 
NAME = 'name' # ident name
OUT_DIR = f'dataset/{NAME}'
DEVICE = 'cuda'

#  folders
os.makedirs(f"{OUT_DIR}/imgs", exist_ok=True)
os.makedirs(f"{OUT_DIR}/track_params", exist_ok=True)

# smirk encoder
smirk_encoder = SmirkEncoder().to(DEVICE)
checkpoint = torch.load('bin/SMIRK_em1.pt')
checkpoint_encoder = {k.replace('smirk_encoder.', ''): v for k, v in checkpoint.items() if 'smirk_encoder' in k}
smirk_encoder.load_state_dict(checkpoint_encoder)
smirk_encoder.eval()

# video
cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"🚀 Processing {frame_count} frames for {NAME}...")

with torch.no_grad():
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret: break

        # rotation fix 
        # ROTATE_90_COUNTERCLOCKWISE if needed
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) 

        # save to imgs
        frame_name = f"{i:05d}"
        cv2.imwrite(f"{OUT_DIR}/imgs/{frame_name}.png", frame)

        # smirk 
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_tensor = cv2.resize(img_rgb, (224, 224))
        img_tensor = torch.tensor(img_tensor).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(DEVICE)

        outputs = smirk_encoder(img_tensor)
        print(f"DEBUG: The keys in outputs are: {outputs.keys()}")

        # save npy as flashavatar expects
        # combine pose_params and jaw_params into one 'pose' vector
        full_pose = torch.cat([outputs['pose_params'], outputs['jaw_params']], dim=-1)

        data_dict = {
            'exp': outputs['expression_params'].cpu().numpy().flatten(),
            'pose': full_pose.cpu().numpy().flatten(),
            'shape': outputs['shape_params'].cpu().numpy().flatten()
        }

        if 'eyelid_params' in outputs:
            data_dict['eyelid'] = outputs['eyelid_params'].cpu().numpy().flatten()
        np.save(f"{OUT_DIR}/track_params/{frame_name}.npy", data_dict)

cap.release()
print(f"\n✅ Done! Check {OUT_DIR} for your data.")
