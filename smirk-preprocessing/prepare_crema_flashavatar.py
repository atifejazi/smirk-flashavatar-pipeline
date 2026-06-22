"""
prepare_crema_flashavatar.py
----------------------------
prepare imgs/, alpha/, and parsing/ folders for a set of CREMA-D sequences
so they match the FlashAvatar dataset format (Obama reference).

what this does: get frames from .flv, chroma key the dark green background into alpha jpgs, run bisenet, and convert masks into a proper parsing folder.

"""

import os
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from tqdm import tqdm


FACE_PARSING_DIR = r'dir' # edit
sys.path.insert(0, FACE_PARSING_DIR)

from models.bisenet import BiSeNet  

SUBJECTS = ['1001', '1005', '1010'] # which crema-d subjects to use. edit as needed.
DATASETS_ROOT = r'pathto\smirk\datasets'

# Build full sequence list from disk (all subject sub-dirs)
def _all_seqs():
    seqs = []
    for subj in SUBJECTS:
        subj_dir = os.path.join(DATASETS_ROOT, subj)
        for name in sorted(os.listdir(subj_dir)):
            if os.path.isdir(os.path.join(subj_dir, name)):
                seqs.append(name)
    return seqs

SEQS = _all_seqs()

FLV_DIR     = r'\\pathto\\VideoFlash'
OUTPUT_BASE = r'\\pathto\\FlashAvatar\\dataset'
WEIGHTS     = r'\\pathto\\face-parsing\weights\resnet34.pt'
TARGET_SIZE = 512

# bisenet class assignments:
# 0=bg, 1=skin, 2=l_brow, 3=r_brow, 4=l_eye, 5=r_eye, 6=eye_g,
# 7=l_ear, 8=r_ear, 9=ear_r, 10=nose, 11=mouth, 12=u_lip, 13=l_lip,
# 14=neck, 15=neck_l, 16=cloth, 17=hair, 18=hat

# head/face region used to mask out background and cloth during training
NECKHEAD_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

# mouth interior andboth lips (higher-weight training region)
MOUTH_CLASSES = {11, 12, 13}


# alpha

def generate_alpha(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Chroma-key the dark-green CREMA-D background.

    The bg is approximately BGR ~[13-37, 71-136, 6-52]
    (HSV ~H=90-100°, S=50-100%, V<55%).
    Returns a single-channel uint8 mask: 0=background, 255=foreground.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([35, 35,  0], dtype=np.uint8)
    upper = np.array([95, 255, 140], dtype=np.uint8)
    bg_mask = cv2.inRange(hsv, lower, upper)

    fg_mask = cv2.bitwise_not(bg_mask)

    # deal with noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return fg_mask


# bisenet

def load_bisenet(device: torch.device) -> torch.nn.Module:
    model = BiSeNet(19, backbone_name='resnet34')
    model.to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()
    return model


_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


@torch.no_grad()
def bisenet_mask(model: torch.nn.Module, img_rgb: np.ndarray,
                 device: torch.device) -> np.ndarray:
    """
    Internally resizes to 512x512, returns (H, W) uint8 class-index mask at original res.
    """
    pil = Image.fromarray(img_rgb).resize((512, 512), Image.BILINEAR)
    t = _TRANSFORM(pil).unsqueeze(0).to(device)
    out = model(t)[0]
    mask512 = out.squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)
    # resize if needed
    if img_rgb.shape[:2] != (512, 512):
        mask = np.array(
            Image.fromarray(mask512).resize(
                (img_rgb.shape[1], img_rgb.shape[0]), Image.NEAREST))
    else:
        mask = mask512
    return mask


#process
def process_sequence(seq_name: str, model: torch.nn.Module,
                     device: torch.device) -> None:
    flv_path = os.path.join(FLV_DIR, seq_name + '.flv')
    if not os.path.exists(flv_path):
        print(f'[SKIP] {flv_path} not found')
        return

    out_dir     = os.path.join(OUTPUT_BASE, seq_name)

    # if imgs/ already has files then count expected frames and skip if done
    imgs_dir_check = os.path.join(out_dir, 'imgs')
    if os.path.isdir(imgs_dir_check):
        existing = len([f for f in os.listdir(imgs_dir_check) if f.endswith('.jpg')])
        if existing > 0:
            cap_check = cv2.VideoCapture(flv_path)
            expected  = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_check.release()
            if existing >= expected:
                print(f'[DONE]  {seq_name} ({existing}/{expected} frames already saved)')
                return
    imgs_dir    = os.path.join(out_dir, 'imgs')
    alpha_dir   = os.path.join(out_dir, 'alpha')
    parsing_dir = os.path.join(out_dir, 'parsing')
    for d in [imgs_dir, alpha_dir, parsing_dir]:
        os.makedirs(d, exist_ok=True)

    cap = cv2.VideoCapture(flv_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'\n[{seq_name}]  {n_frames} frames  ->  {out_dir}')

    for i in tqdm(range(n_frames), desc=f'  {seq_name}'):
        ret, frame_bgr = cap.read()
        if not ret:
            break

        name = f'{i:05d}'

        # resize to 512x512
        frame_resized = cv2.resize(frame_bgr, (TARGET_SIZE, TARGET_SIZE))

        # jpg
        cv2.imwrite(os.path.join(imgs_dir, name + '.jpg'), frame_resized,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        # alpha save
        alpha = generate_alpha(frame_resized)
        cv2.imwrite(os.path.join(alpha_dir, name + '.jpg'), alpha,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        #  bisenet
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        raw_mask = bisenet_mask(model, frame_rgb, device)

        # binary flashav masks
        neckhead = np.isin(raw_mask, list(NECKHEAD_CLASSES)).astype(np.uint8) * 255
        mouth    = np.isin(raw_mask, list(MOUTH_CLASSES)).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(parsing_dir, name + '_neckhead.png'), neckhead)
        cv2.imwrite(os.path.join(parsing_dir, name + '_mouth.png'),    mouth)

    cap.release()


# main

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Sequences to process: {len(SEQS)}  (subjects: {SUBJECTS})')
    print(f'Loading BiSeNet weights from {WEIGHTS}')
    model = load_bisenet(device)

    for idx, seq in enumerate(SEQS, 1):
        print(f'[{idx}/{len(SEQS)}]', end=' ')
        process_sequence(seq, model, device)

    print('\n=== Done. Output at:', OUTPUT_BASE)
