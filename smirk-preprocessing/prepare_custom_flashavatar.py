"""
prepare_custom_flashavatar.py:
in the wild non green screen clips: center-crop to square (we will not smush together), RVM alpha, BiSeNet parsing.

Do not use prepare_crema_flashavatar.py for these since its green-screen chroma key produces all-white alpha on real backgrounds. (dafoe fix here)

"""

import glob
import os
import subprocess
import sys

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

# EDIT PATHS 
FACE_PARSING_DIR = 'path/to/face-parsing'
RVM_DIR = 'path/to/RobustVideoMatting'
RVM_CHECKPOINT = os.path.join(RVM_DIR, 'checkpoints', 'rvm_mobilenetv3.pth')
INPUT_DIR = 'path/to/input-videos'
OUTPUT_BASE = 'path/to/FlashAvatar-code/dataset'
WEIGHTS = os.path.join(FACE_PARSING_DIR, 'weights', 'resnet34.pt')
TARGET_SIZE = 512

# OUTPUT
SEQS = ['MyClipName']
VIDEO_SOURCES = {'MyClipName': 'my_source_video'}

sys.path.insert(0, FACE_PARSING_DIR)
from models.bisenet import BiSeNet

NECKHEAD_CLASSES = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
MOUTH_CLASSES = {11, 12, 13}


def center_crop_resize(frame_bgr: np.ndarray, size: int = TARGET_SIZE) -> np.ndarray:
    
    h, w = frame_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = frame_bgr[y0:y0 + side, x0:x0 + side]
    if side != size:
        cropped = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
    return cropped


def resolve_video(seq_name: str):
    stem = VIDEO_SOURCES.get(seq_name, seq_name)
    for ext in ('.flv', '.mp4', '.mov'):
        path = os.path.join(INPUT_DIR, stem + ext)
        if os.path.exists(path):
            return path
    return None


def export_cropped_video(src_path: str, dst_path: str) -> int:
    cap = cv2.VideoCapture(src_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
    writer = cv2.VideoWriter(
        dst_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (TARGET_SIZE, TARGET_SIZE))
    count = 0
    for _ in range(n):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(center_crop_resize(frame))
        count += 1
    cap.release()
    writer.release()
    return count


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
def bisenet_mask(model, img_rgb, device):
    pil = Image.fromarray(img_rgb)
    t = _TRANSFORM(pil).unsqueeze(0).to(device)
    return model(t)[0].squeeze(0).cpu().numpy().argmax(0).astype(np.uint8)


def run_rvm_on_imgs(imgs_dir: str, alpha_dir: str) -> None:
    """Run Robust Video Matting on imgs/ and write alpha/*.jpg."""
    tmp_alpha = os.path.join(alpha_dir, '_rvm_tmp')
    os.makedirs(tmp_alpha, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(RVM_DIR, 'inference.py'),
        '--variant', 'mobilenetv3',
        '--checkpoint', RVM_CHECKPOINT,
        '--device', 'cuda' if torch.cuda.is_available() else 'cpu',
        '--input-source', imgs_dir,
        '--output-type', 'png_sequence',
        '--output-alpha', tmp_alpha,
        '--seq-chunk', '4',
        '--num-workers', '2',
    ]
    env = os.environ.copy()
    env['PYTHONPATH'] = RVM_DIR + (':' + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    print(f'  Running RVM on {imgs_dir} ...')
    subprocess.run(cmd, check=True, cwd=RVM_DIR, env=env)

    for i, path in enumerate(sorted(glob.glob(os.path.join(tmp_alpha, '*.png')))):
        alpha = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(
            os.path.join(alpha_dir, f'{i:05d}.jpg'), alpha,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
    for f in glob.glob(os.path.join(tmp_alpha, '*')):
        os.remove(f)
    os.rmdir(tmp_alpha)


def process_sequence(seq_name: str, model, device) -> None:
    video_path = resolve_video(seq_name)
    if video_path is None:
        print(f'[SKIP] no video for {seq_name}')
        return

    subject_id = seq_name[:4]
    out_dir = os.path.join(OUTPUT_BASE, subject_id, seq_name)
    imgs_dir = os.path.join(out_dir, 'imgs')
    alpha_dir = os.path.join(out_dir, 'alpha')
    parsing_dir = os.path.join(out_dir, 'parsing')
    for d in [imgs_dir, alpha_dir, parsing_dir]:
        os.makedirs(d, exist_ok=True)

    cropped_vid = os.path.join(INPUT_DIR, seq_name + '.mp4')
    if not os.path.exists(cropped_vid):
        print(f'  Writing cropped video -> {cropped_vid}')
        export_cropped_video(video_path, cropped_vid)

    cap = cv2.VideoCapture(video_path)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'\n[{seq_name}] {n_frames} frames (center crop) -> {out_dir}')

    for i in tqdm(range(n_frames), desc=f'  {seq_name} imgs+parsing'):
        ret, frame_bgr = cap.read()
        if not ret:
            break
        name = f'{i:05d}'
        frame_sq = center_crop_resize(frame_bgr)
        cv2.imwrite(
            os.path.join(imgs_dir, name + '.jpg'), frame_sq,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        frame_rgb = cv2.cvtColor(frame_sq, cv2.COLOR_BGR2RGB)
        raw_mask = bisenet_mask(model, frame_rgb, device)
        neckhead = np.isin(raw_mask, list(NECKHEAD_CLASSES)).astype(np.uint8) * 255
        mouth = np.isin(raw_mask, list(MOUTH_CLASSES)).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(parsing_dir, name + '_neckhead.png'), neckhead)
        cv2.imwrite(os.path.join(parsing_dir, name + '_mouth.png'), mouth)
    cap.release()

    run_rvm_on_imgs(imgs_dir, alpha_dir)
    sample = cv2.imread(os.path.join(alpha_dir, '00010.jpg'), cv2.IMREAD_GRAYSCALE)
    fg = (sample > 127).mean() * 100
    print(f'  RVM alpha frame 10: {fg:.1f}% foreground (expect well below 100%)')


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    model = load_bisenet(device)
    for seq in SEQS:
        process_sequence(seq, model, device)
    print('\n=== Done. Output at:', OUTPUT_BASE)
