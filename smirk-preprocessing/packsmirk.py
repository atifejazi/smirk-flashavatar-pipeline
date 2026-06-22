import torch
import numpy as np
import os
from glob import glob

# paths (CHANGE  IF NEEDED)
SMIRK_NPY_DIR = 'dataset/name/track_params/'
SAVE_PATH = 'metrical-tracker/output/name/track_params.pt'
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

#  .npy files
npy_files = sorted(glob(os.path.join(SMIRK_NPY_DIR, '*.npy')))

all_params = {
    'exp': [],
    'pose': [],
    'shape': []
}

for f in npy_files:
    data = np.load(f, allow_pickle=True).item()
    all_params['exp'].append(torch.tensor(data['exp']))
    all_params['pose'].append(torch.tensor(data['pose']))
    all_params['shape'].append(torch.tensor(data['shape']))

final_data = {
    'exp': torch.stack(all_params['exp']).squeeze(),
    'pose': torch.stack(all_params['pose']).squeeze(),
    'shape': torch.stack(all_params['shape']).squeeze()
}

torch.save(final_data, SAVE_PATH)
print(f"Packed {len(npy_files)} frames into {SAVE_PATH}")
