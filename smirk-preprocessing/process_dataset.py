import os
import glob
import shutil
import subprocess
from tqdm import tqdm

# i/o stuff
INPUT_DIR = "VideoFlash"              # .flv file input
SMIRK_OUT_DIR = "results"           
FINAL_DATASET_DIR = "datasets"        # outp
CHECKPOINT_PATH = "pretrained_models/SMIRK_em1.pt"

def process_and_organize_dataset():
    os.makedirs(SMIRK_OUT_DIR, exist_ok=True)
    os.makedirs(FINAL_DATASET_DIR, exist_ok=True)

    video_files = glob.glob(os.path.join(INPUT_DIR, "*.flv"))
    
    if not video_files:
        print(f"No .flv files found in {INPUT_DIR}. Please check the folder path.")
        return

    # progress bar loop
    
    for video_path in tqdm(video_files, desc="Processing CREMA-D"):
        # ex 1001_DFA_ANG_XX from VideoFlash/1001_DFA_ANG_XX.flv
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        # ex 1001
        subject_id = base_name[:4] 
        
        target_dir = os.path.join(FINAL_DATASET_DIR, subject_id, base_name)
        
        # check if exists
        if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
            continue
            
        # clean
        for old_file in glob.glob(os.path.join(SMIRK_OUT_DIR, "*")):
            os.remove(old_file)

        # 2. SMIRK
        command = [
            "python", "demo_video.py",
            "--input_path", video_path,
            "--out_path", SMIRK_OUT_DIR,
            "--checkpoint", CHECKPOINT_PATH,
            "--crop"
        ]
        
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            print(f"\nError processing {base_name}. Skipping to next video.")
            continue

        os.makedirs(target_dir, exist_ok=True)
        
        generated_files = glob.glob(os.path.join(SMIRK_OUT_DIR, "*"))
        
        for file_path in generated_files:
            file_name = os.path.basename(file_path)
            
            # location of mesh
            if "template_mesh" in file_name:
                dest_path = os.path.join(FINAL_DATASET_DIR, subject_id, file_name)
            
            elif file_name.endswith('.mp4'):
                dest_path = os.path.join(target_dir, f"{base_name}.mp4")
                
            else:
                dest_path = os.path.join(target_dir, file_name)
                
            shutil.move(file_path, dest_path)

if __name__ == "__main__":
    process_and_organize_dataset()
