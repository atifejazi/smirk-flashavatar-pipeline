# Video to FlashAvatar Pipeline using SMIRK

The goal of this repo is to go from a monocular video to a face avatar using FlashAvatar, including a video of rendered Gaussians.

There are two primary phases to this pipeline:

1. **Preprocessing**: turn the video into FlashAvatar folders (**`imgs/`**, **`alpha/`**, **`parsing/`**) and get *tracking data* from **SMIRK**, then convert that into *`.frame`* files.
2. **FlashAvatar**: train then render with `test.py`.

FlashAvatar step must be done using WSL/Linux. Although it is fine to perform the SMIRK step using Windows, it is recommended to do so in WSL/Linux which is what this repo assumes.

**Recmomended layout:** (`main` is the overarching parent folder):

```
main/
  smirk-flashavatar-pipeline/   # THIS repo, which includes custom scripts and patches and this README
  FlashAvatar-code/             # cloned from flashavatar's git repo, and then apply patches and custom scripts
  smirk/                        # cloned from smirk's git repo, and then add the smirk-preprocessing folder from our pipeline repo into here
  face-parsing/                 # cloned from BiSeNet's git repo
  RobustVideoMatting/           # cloned from RVM's git repo, for non crema-d 
  input-videos/                 # video input into the pipeline. not required but paths are set for this kind of layout.
  pytorch3d-build/              # pytorch3d clone (optional, good idea if RAM constraints)
```

Edit path constants at the top of each preprocessing script before running if needed.
---

## Conda environment setup

Use two separate env for SMIRK and FLashAvatar.

### FlashAvatar (WSL)

Clone or `cd` into the [FlashAvatar](https://github.com/USTC3DV/FlashAvatar-code) repo first. Ensure submodules are cloned:
```
git clone https://github.com/USTC3DV/FlashAvatar-code.git --recursive
```

#### Option A: RTX 3090 / CUDA 11.6

Follow the README of FlashAvatar on their GitHub page. Use this if you are on an older GPU and the default env creates without issues.

```bash
cd path/to/FlashAvatar-code
git submodule update --init --recursive

conda env create -f environment.yml
conda activate FlashAvatar
```

`environment.yml` already lists the two CUDA extensions under `pip:`; if import fails, reinstall them explicitly:

```bash
pip install submodules/diff-gaussian-rasterization submodules/simple-knn
```

Install **PyTorch3D** (README):

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -c bottler nvidiacub
conda install pytorch3d -c pytorch3d
```

#### Option B: RTX 50-series / CUDA 12.8 (custom lab setup)

The given `environment.yml` from FlashAvatar does not match 50-series GPU and requires modifications in the versions of CUDA, Python, etc. Follow below:

```bash
cd path/to/FlashAvatar-code
git submodule update --init --recursive

conda create -n FlashAvatar python=3.10 -y
conda activate FlashAvatar
pip install --upgrade pip setuptools wheel

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

pip install "numpy<2"   # chumpy / legacy deps break on numpy 2.x
pip install scipy chumpy scikit-image opencv-python ninja lpips loguru plyfile tqdm yacs

conda install -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8
conda install -y -c conda-forge gxx_linux-64=12.4 gcc_linux-64=12.4

export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CONDA_PREFIX/bin:$CONDA_PREFIX/nvvm/bin:$CUDA_HOME/bin:$PATH"
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
pip install --no-build-isolation submodules/diff-gaussian-rasterization submodules/simple-knn


git clone --depth 1 --branch stable https://github.com/facebookresearch/pytorch3d.git path/to/pytorch3d-build
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/nvvm/bin:$CONDA_PREFIX/bin:$PATH"
export CPATH="$CUDA_HOME/include:$(find $CONDA_PREFIX/lib/python3.10/site-packages/nvidia -type d -name include | tr '\n' ':')"
export LIBRARY_PATH="$(find $CONDA_PREFIX/lib/python3.10/site-packages/nvidia -type d -name lib | tr '\n' ':')$CONDA_PREFIX/lib"
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
pip install --no-build-isolation path/to/pytorch3d-build

```



**Note on chumpy/pickle failure:** if `generic_model.pkl` fails with pickle/`numpy` errors on chumpy 0.70, patch `$CONDA_PREFIX/lib/python3.10/site-packages/chumpy/__init__.py` so `from numpy import bool, int, float, complex, object, unicode, str, nan, inf` becomes `from numpy import bool_, int_, float_, complex_, object_, nan, inf` (drop `unicode` / bare `str`). Then re-run after `pip install chumpy`.

If the rasterizer step still fails, reinstall torch for your GPU, then rerun the two `pip install --no-build-isolation submodules/...` lines.

If **`simple_knn`** fails with `FLT_MAX is undefined`, add `#include <float.h>` as the first line of `submodules/simple-knn/simple_knn.cu`, then rerun the `simple-knn` install.

Do NOT run `conda install pytorch3d` in this env after installing CUDA torch as it can replace torch with a CPU build. Build from source.

Additional/alt PyTorch3D install notes: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

#### Download FLAME model file 

FlashAvatar expects the licensed FLAME geometry at **`flame/generic_model.pkl`** (see `flame/mica_flame_config.py`). It is **not** in git.

1. Register or login: https://flame.is.tue.mpg.de/
2. Download **`generic_model.pkl`** (FLAME 2020)
3. Place it at **`flame/generic_model.pkl`** inside the FlashAvatar repo

FlashAvatar training also needs **`flame/FLAME_masks/FLAME_masks.pkl`**. If you don't have it already from previous projects, note that they come from the same download page as above (segmentation / vertex masks).



### SMIRK (WSL or Windows)

First, clone the [SMIRK Github Repo](https://github.com/georgeretsi/smirk)

**Option A: RTX 50-series / CUDA 12.8**: Follow these steps exactly.

```bash
conda create -n smirk python=3.9 -y
conda activate smirk
pip install --upgrade pip setuptools wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install albumentations==1.3.0 mediapipe==0.10.10 "numpy<2" omegaconf==2.3.0 \
  opencv_contrib_python==4.9.0.80 opencv_python==4.9.0.80 opencv_python_headless==4.9.0.80 \
  scikit_learn==1.3.2 scikit-image==0.22.0 timm==0.9.16 tqdm==4.66.2 chumpy==0.70 \
  fvcore iopath pytorch_lightning==2.2.1 gdown
conda install -y -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8 cmake ninja
git clone --depth 1 --branch stable https://github.com/facebookresearch/pytorch3d.git path/to/pytorch3d-build
export CUDA_HOME="$CONDA_PREFIX/targets/x86_64-linux"
export PATH="$CUDA_HOME/bin:$CONDA_PREFIX/nvvm/bin:$CONDA_PREFIX/bin:$PATH"
unset NVCC_PREPEND_FLAGS
export CPATH="$CUDA_HOME/include:$(find $CONDA_PREFIX/lib/python3.9/site-packages/nvidia -type d -name include | tr '\n' ':')"
export LIBRARY_PATH="$(find $CONDA_PREFIX/lib/python3.9/site-packages/nvidia -type d -name lib | tr '\n' ':')$CONDA_PREFIX/lib"
export TORCH_CUDA_ARCH_LIST="12.0"
export FORCE_CUDA=1
MAX_JOBS=1 pip install --no-build-isolation path/to/pytorch3d-build
```

**Additional note:** keep `"numpy<2"`, and if `conda install av` breaks `scikit-image`, reinstall with `pip install scikit-image==0.22.0`. 

**Option B: Older GPUs (RTX 30/40 -series):** follow SMIRK install instructions from their GitHub page instead instead from [SMIRK README](https://github.com/georgeretsi/smirk#installation).

**Download Model assets** (once per machine, from the SMIRK repo root):

```bash
mkdir -p pretrained_models assets/FLAME2020
gdown --id 1T65uEd9dVLHgVw5KiUYL66NUee-MCzoE -O pretrained_models/SMIRK_em1.pt
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task \
  -O assets/face_landmarker.task
```
Get FLAME 2020 generic model (just like earlier) from `https://flame.is.tue.mpg.de/` and place `generic_model.pkl` under `assets/FLAME2020/`


### IMPORTANT: Use the modified/custom scripts from this repo
The default scripts from the installed FlashAvatar and SMIRK are not sufficient. THIS repo comes with scripts you are supposed to use.
| File / Folder | Type | Description |
| :--- | :--- | :--- |
| **`smirk-preprocessing/`** | Directory | Copy this directly into the SMIRK repo. Includes **`prepare_crema_flashavatar.py`** (CREMA green-screen) and **`prepare_custom_flashavatar.py`** (in-the-wild: center crop + RVM alpha). |
| **`flashavatar/`** | Directory | Patched FlashAvatar files: `package.py`, `train.py`, `test.py`, `flame/flame_mica.py`, `scene/`, `src/deform_model.py`, and `tools/` (see Step 2). |
| **`smirk_requirements_pip.txt`** | File | Optional `pip` configuration pins for SMIRK setup. |
| **`flashavatar_environment.yml`** | File | Optional `conda` environment export for FlashAvatar (Lab setup). |
| **`README.md`** | File | The core instruction manual for the workflow pipeline. |

#### Step 1: Copy SMIRK scripts
Clone this repo, and from this repo, copy the folder into SMIRK repo root.
```
cp -r path/to/<pipeline-repo>/smirk-preprocessing path/to/smirk/
```
Note: Due to how the paths are, when running `demo_video.py`, `process_for_flashavatar.py`, `export_verts.py`, or `process_dataset`, make sure you're currently in the SMIRK root directory (NOT the `smirk-preprocessing folder`) and run it like `python smirk-preprocessing/<script>.py ...`. If you get `ModuleNotFoundError: No module named 'src'`, prefix with `PYTHONPATH=path/to/smirk` or `export PYTHONPATH=$(pwd)` from the SMIRK root. The code to run them will be explained in detail and step by step below.

#### Step 2: Copy FlashAvatar scripts
Either merge from `flashavatar/` in this pipeline repo, or copy individual files into your FlashAvatar clone:
```bash
# From pipeline repo root — copies custom root script + tools/
mkdir -p path/to/FlashAvatar-code/tools
cp path/to/<pipeline-repo>/flashavatar/package.py path/to/FlashAvatar-code/
cp path/to/<pipeline-repo>/flashavatar/train.py path/to/FlashAvatar-code/
cp path/to/<pipeline-repo>/flashavatar/test.py path/to/FlashAvatar-code/
cp path/to/<pipeline-repo>/flashavatar/flame/flame_mica.py path/to/FlashAvatar-code/flame/
cp path/to/<pipeline-repo>/flashavatar/scene/*.py path/to/FlashAvatar-code/scene/
cp path/to/<pipeline-repo>/flashavatar/src/deform_model.py path/to/FlashAvatar-code/src/
cp path/to/<pipeline-repo>/flashavatar/tools/fit_flame_exp.py path/to/FlashAvatar-code/tools/
cp path/to/<pipeline-repo>/flashavatar/tools/inspect_pt.py path/to/FlashAvatar-code/tools/
cp path/to/<pipeline-repo>/flashavatar/tools/prepare_crema_clip.py path/to/FlashAvatar-code/tools/
cp path/to/<pipeline-repo>/flashavatar/tools/run_identity.sh path/to/FlashAvatar-code/tools/
cp path/to/<pipeline-repo>/flashavatar/tools/run_all_crema.sh path/to/FlashAvatar-code/tools/
chmod +x path/to/FlashAvatar-code/tools/*.sh
```

Note: `flame_mica.py` path is **`flashavatar/flame/flame_mica.py`**, not the repo root.

Furthermore, the FlashAvatar `train.py` / `test.py` / `scene/` do not accept this pipeline's `--downscale`, `--full_iters`, or `--frame_delta` flags so the copies above are required.


---

## What you need locally for FlashAvatar:

FlashAvatar expects, for each identity **`idname`**:

**Images and masks** (use BiSeNet in this pipeline; **CREMA-D** uses green-screen chroma key, **custom clips** use RVM — see below):

**Dataset layout:** preprocessing writes nested folders and training uses a flat **`idname`** via symlink from `prepare_crema_clip.py`:

```
dataset/<subject_id>/<idname>/     # example: dataset/1015/1015_DFA_ANG_XX/
  imgs/          # video frames (512×512 .jpg for this pipeline)
  alpha/         # alpha masks (same frame indices)
  parsing/       # mouth / neckhead masks

dataset/<idname>  -> symlink to dataset/<subject_id>/<idname>/   # created by prepare_crema_clip.py
```

**Per-frame FLAME tracking** for FlashAvatar’s loader:

```
metrical-tracker/output/<idname>/checkpoint/
  00000.frame
  00001.frame
  ...
```

Note: the .frame files are not produced by raw SMIRK but need to be built using the `package.py` custom script (or alternatively, by `tools/prepare_crema_clip.py` if you're using CREMA-D dataset which will be explained in detail below). 

---

## Preprocessing step

Start from a video (or a folder of frames). End with the aforementioned layout.

### `imgs/`, `alpha/`, `parsing/`

Goal: Extract frames into **`imgs/`**. Build **`alpha/`** (one mask per frame, same numbering as **`imgs/`**) and **`parsing/`** (face parsing masks, e.g. mouth / neck+head).

Note: If the subject is not on a clean background, use [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) (RVM) or similar: export a video or frames, then turn that into **`alpha/NNNNN.png`** aligned with **`imgs/`**.
**`smirk-preprocessing/alphasegment.py`** splits an alpha **`.mp4`** into per-frame **`alpha/`** (edit **`ALPHA_VIDEO_PATH`** and **`OUT_DIR`** at the top, then):

```bash
conda activate smirk   # or whichever env has opencv
cd path/to/smirk-preprocessing
python alphasegment.py
```


For a single custom video, **`process_for_flashavatar.py`** saves **`imgs/`** and per-frame **`track_params/`** **`.npy`** (edit **`VIDEO_PATH`**, **`NAME`**, and **`OUT_DIR`** at the top of the script. Set **`OUT_DIR`** under your FlashAvatar repo, like for example, `path/to/FlashAvatar-code/dataset/<idname>`, so **`imgs/`** land where BiSeNet and training expect them). Then:

```bash
conda activate smirk
cd path/to/smirk
python smirk-preprocessing/process_for_flashavatar.py
python smirk-preprocessing/packsmirk.py   # edit SMIRK_NPY_DIR and SAVE_PATH at top first
```

### BiSeNet face parsing (`face-parsing` env)

These are the steps for **face parsing**. FlashAvatar uses **`parsing/`** masks to up-weight the **mouth region** during training. Without them, training still runs but mouth mask defaults to zeros. For each frame, we see:
```
dataset/
└── parsing/
    ├── 00000_mouth.png       # Binary (0/255) - Mouth interior + upper/lower lip
    ├── 00000_neckhead.png    # Binary (0/255) - Head/neck/face region
    └── 00001_mouth.png       # Sequential frames...
```

Note that frame indices must match.

This pipeline uses **BiSeNet + ResNet34 weights**. Make sure to use a third conda env (e.g. `face-parsing`) and do not install into `smirk` or `FlashAvatar`. Execute in terminal as below:



#### 1. Create the `face-parsing` environment

```bash
conda create -n face-parsing python=3.10 -y
conda activate face-parsing
pip install --upgrade pip

git clone https://github.com/yakhyo/face-parsing.git path/to/face-parsing
cd path/to/face-parsing
pip install -r requirements.txt
pip install "numpy<2"   # if cv2 fails with numpy.core.multiarray error

mkdir -p weights
wget -O weights/resnet34.pt \
  https://github.com/yakhyo/face-parsing/releases/download/weights/resnet34.pt
```

#### Generic video (you already have dataset/<idname>/imgs/)
If  the frames are already in dataset/<idname>/imgs/ (from process_for_flashavatar.py, etc.), run BiSeNet on that folder, then convert raw class masks to FlashAvatar format.

Step 1: raw segmentation:
```bash
conda activate face-parsing
cd path/to/face-parsing

python inference.py \
  --model resnet34 \
  --weight ./weights/resnet34.pt \
  --input path/to/FlashAvatar-code/dataset/<idname>/imgs \
  --output /tmp/bisenet_raw/<idname>
```

On **RTX 50-series**, if GPU inference fails with `no kernel image is available for execution on the device`, rerun with CPU: `CUDA_VISIBLE_DEVICES="" python inference.py ...` (same args). 

Step 2: convert to `parsing/`: (note this step is not necessary if you use CREMA_D as there is a custom script for it explained later).
```
conda activate face-parsing   # needs numpy, opencv, pillow

python - <<'PY'
import os, glob
import cv2
import numpy as np

ID = "<idname>"
RAW_DIR = f"/tmp/bisenet_raw/{ID}/resnet34"   # saves under output/model/
OUT_DIR = f"path/to/FlashAvatar-code/dataset/{ID}/parsing"
os.makedirs(OUT_DIR, exist_ok=True)

MOUTH = {11, 12, 13}
NECKHEAD = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}

for raw_path in sorted(glob.glob(os.path.join(RAW_DIR, "*_raw.png"))):
    stem = os.path.basename(raw_path).replace("_raw.png", "")
    mask = cv2.imread(raw_path, cv2.IMREAD_UNCHANGED)
    mouth = (np.isin(mask, list(MOUTH)).astype(np.uint8) * 255)
    neckhead = (np.isin(mask, list(NECKHEAD)).astype(np.uint8) * 255)
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_mouth.png"), mouth)
    cv2.imwrite(os.path.join(OUT_DIR, f"{stem}_neckhead.png"), neckhead)
    print(stem)
print("Wrote parsing masks to", OUT_DIR)
PY
```


### SMIRK: parameters + mesh

Run **SMIRK** on the same video (or on extracted frames) so you get the followign for each clip:

- **`frame_XXXX.npy`** : per-frame SMIRK outputs
- **`<name>_mesh_sequence.npy`** : vertex sequence **(N, 5023, 3)** for fitting

We use the modified **`demo_video.py`** in **`smirk-preprocessing/`** instead of the one given by default.

```bash
conda activate smirk
cd path/to/smirk
python smirk-preprocessing/demo_video.py \
  --input_path path/to/video.mp4 \
  --out_path results \
  --checkpoint pretrained_models/SMIRK_em1.pt \
  --crop
```

This writes **`results/frame_XXXX.npy`**, **`results/<video_name>_mesh_sequence.npy`**, and a preview video under **`results/`**.
Note: we are using the mesh file since SMIRK's expression is 50-dim while FlashAvatar has a 100-dim FLAME expression. We fit from the mesh and build single verts file from per-frame track params. Execute as below.

```bash
conda activate smirk
cd path/to/smirk
python smirk-preprocessing/export_verts.py \
  --track_dir dataset/<idname>/track_params \
  --out_verts path/to/verts.npy
```

### CREMA-D

CREMA-D is a massive monocular emotional video dataset which this pipeline was intended for. For the purposes of this lab, the private `dataset` Google Drive folder holds the already pre-processed data with SMIRK and other scripts. It just needs to go through the preprocessing through BiSeNet step and the FlashAvatar step. However, if access to that folder is not possible, then please observe below.

If you are working with CREMA-D (green-screen **`.flv`** or **`.mp4`** clips), batch paths and layout helpers live under **`smirk-preprocessing/`**. Edit **`FLV_DIR`**, **`OUTPUT_BASE`**, **`FACE_PARSING_DIR`**, and **`SUBJECTS`** at the top of **`prepare_crema_flashavatar.py`**, then:

```bash
conda activate smirk
cd path/to/smirk

# SMIRK on every .flv in VideoFlash/
python smirk-preprocessing/process_dataset.py

# imgs/, alpha/, parsing/ — needs torch + BiSeNet (NOT the face-parsing-only env)
conda activate FlashAvatar
python smirk-preprocessing/prepare_crema_flashavatar.py
```

**NOTE:** CREMA output layout writes **`dataset/<subject_id>/<seq_name>/`** (example: `dataset/1015/1015_DFA_ANG_XX/`). It supports **`.flv`** and **`.mp4`** in **`FLV_DIR`**. Do **not** use this script on in-the-wild backgrounds as chroma key assumes CREMA's dark green screen.

On the **FlashAvatar** side (WSL, **`FlashAvatar`** env):

```bash
conda activate FlashAvatar
cd path/to/FlashAvatar-code

python tools/prepare_crema_clip.py \
  --clip_dir dataset/1015/1015_DFA_ANG_XX \
  --idname 1015_DFA_ANG_XX

# many clips:
bash tools/run_all_crema.sh
```

`prepare_crema_clip.py` symlinks **`dataset/<idname>`** → the nested clip folder so **`run_identity.sh`** can use the flat id.

### Custom / in-the-wild video (non-green-screen)

**NOTE:** Use **`prepare_custom_flashavatar.py`**, not **`prepare_crema_flashavatar.py`**. Squashing 16:9 → 512×512 without cropping might hurt quality. Alpha comes from **[Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting)** (RVM), not chroma key like with CREMA-D processing.

RVM setup (FlashAvatar env — RVM runs here during preprocessing):

```bash
conda activate FlashAvatar
git clone https://github.com/PeterL1n/RobustVideoMatting path/to/RobustVideoMatting
wget -P path/to/RobustVideoMatting/checkpoints \
  https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3.pth
conda install -y -c conda-forge av pims
```

Edit **`INPUT_DIR`**, **`OUTPUT_BASE`**, **`FACE_PARSING_DIR`**, **`RVM_DIR`**, **`SEQS`**, and **`VIDEO_SOURCES`** at the top of **`prepare_custom_flashavatar.py`**, then:

```bash
conda activate FlashAvatar
cd path/to/smirk
python smirk-preprocessing/prepare_custom_flashavatar.py
```

The script also writes a **center-cropped proxy** **`input-videos/<idname>.mp4`**. Run SMIRK on **that** file so tracking matches **`imgs/`** geometry:

```bash
conda activate smirk
cd path/to/smirk
python smirk-preprocessing/demo_video.py \
  --input_path path/to/input-videos/<idname>.mp4 \
  --out_path results/<idname> \
  --checkpoint pretrained_models/SMIRK_em1.pt \
  --crop
```

Copy **`frame_*.npy`** and **`*_mesh_sequence.npy`** into **`dataset/<subject_id>/<idname>/`**, then run **`prepare_crema_clip.py`** and **`run_identity.sh`** as for CREMA (same **`--clip_dir`** / **`--idname`** pattern).

---

## Build `.frame` files (WSL, FlashAvatar env)

```bash
conda activate FlashAvatar
cd path/to/FlashAvatar-code
```

**Goal:** (single **`track_params.pt`** + optional **`verts.npy`**):

1. Fit FLAME expression from verts:

```bash
python tools/fit_flame_exp.py \
  --verts_file path/to/verts.npy \
  --pt_file track_params.pt \
  --out_pt track_params_fitted.pt
```

2. Package into `.frame` files (match **`--img_size`** to your **`imgs/`** width and height. Check with `python -c "import cv2; print(cv2.imread('dataset/<id>/imgs/00000.png').shape[1::-1])"` for PNG, or `.jpg` for CREMA):

```bash
python package.py \
  --idname MyID \
  --pt_file track_params_fitted.pt \
  --img_size 720 1280 \
  --focal 1200
```

(Example **`720 1280`** = width × height for a rotated phone clip; CREMA-D **`imgs/`** are typically **`512 512`**.)

**CREMA-D clip** (folder already has **`frame_*.npy`**, **`*_mesh_sequence.npy`**, **`imgs/`**, **`alpha/`**, **`parsing/`** — see CREMA / custom sections above):

```bash
python tools/prepare_crema_clip.py \
  --clip_dir dataset/1015/1015_DFA_ANG_XX \
  --idname 1015_DFA_ANG_XX
```

That writes **`metrical-tracker/output/<idname>/checkpoint/*.frame`** and symlinks **`dataset/<idname>`** → the clip folder.

---

## Train and render 

```bash
conda activate FlashAvatar
cd path/to/FlashAvatar-code
```

**Train + test** (use the flat **`idname`**, not the nested path — e.g. **`1015_DFA_ANG_XX`**, not **`1015/1015_DFA_ANG_XX`**):

```bash
./tools/run_identity.sh 1015_DFA_ANG_XX 0.65 25000 both 0
```

Arguments: **`id`**, **`downscale`**, **`full_iters`**, **`both` / `train_only` / `test_only`**, **`frame_delta`**.  

Pick the checkpoint that exists: saves every 5000 steps, like `chkpnt20000.pth`, `chkpnt25000.pth`. Alternatively, use **`run_identity.sh`**, which loads the **latest** `chkpnt*.pth`):

```bash
python train.py \
  --idname MyClipName \
  --downscale 0.65 \
  --full_iters 25000 \
  --frame_delta 0 \
  --camera_z 1.0 \
  --force_camera_z

python test.py \
  --idname MyClipName \
  --downscale 0.65 \
  --frame_delta 0 \
  --camera_z 1.0 \
  --force_camera_z \
  --render_all \
  --checkpoint dataset/MyClipName/log/ckpt/chkpnt25000.pth
```

Resume training from a checkpoint:

```bash
python train.py --idname MyClipName --downscale 0.65 --full_iters 25000 \
  --frame_delta 0 --camera_z 1.0 --force_camera_z \
  --start_checkpoint dataset/MyClipName/log/ckpt/chkpnt10000.pth
```

Test `downscale` must match train.

**Outputs:**

- Checkpoints: **`dataset/<idname>/log/ckpt/chkpnt*.pth`** (written every **5000** iterations during training)
- Video: **`dataset/<idname>/log/test.avi`** (left is ground truth, right is render); **`run_identity.sh`** also writes **`test.mp4`** if **`ffmpeg`** is installed

