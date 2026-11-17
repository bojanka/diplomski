import sys
sys.path.insert(0, r"D:\Python_packages")

import os
import subprocess

os.environ["nnUNet_raw"] = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\data\nnunet"
os.environ["nnUNet_preprocessed"] = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\data\preprocessed"
os.environ["nnUNet_results"] = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\results"
os.environ["nnUNet_n_proc_DA"] = "2"

result = subprocess.run([
    "nnUNetv2_train",
    "1",
    "3d_lowres",
    "0",
    "--npz",
    "-num_gpus", "1",
    "--c"
], env=os.environ, capture_output=False)