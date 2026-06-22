import os
import shutil
import glob

# test outputs 
smirk_output_dir = "results" 
# final folder
final_dataset_dir = "datasets" 

def organize_smirk_output(video_filename):
    # cop base name
    base_name = os.path.splitext(video_filename)[0]
    
    # subject ID (cremad)
    subject_id = base_name[:4]
    
    # directory
    target_dir = os.path.join(final_dataset_dir, subject_id, base_name)
    os.makedirs(target_dir, exist_ok=True)
    
    mp4_file = os.path.join(smirk_output_dir, f"{base_name}.mp4")
    if os.path.exists(mp4_file):
        shutil.move(mp4_file, os.path.join(target_dir, f"{base_name}.mp4"))
        
    npy_files = glob.glob(os.path.join(smirk_output_dir, "frame_*.npy"))
    for npy in npy_files:
        filename = os.path.basename(npy)
        shutil.move(npy, os.path.join(target_dir, filename))
        
    print(f"Successfully organized {base_name} into {target_dir}")

organize_smirk_output("1001_DFA_ANG_XX")