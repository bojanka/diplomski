import os
import subprocess

# Postavi environment varijable za ovaj proces
os.environ["nnUNet_raw"] = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\data\nnunet"
os.environ["nnUNet_preprocessed"] = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\data\preprocessed"
os.environ["nnUNet_results"] = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\results"

# Provera
print("nnUNet_raw:", os.environ["nnUNet_raw"])
print("nnUNet_preprocessed:", os.environ["nnUNet_preprocessed"])
print("nnUNet_results:", os.environ["nnUNet_results"])

# Pokreni plan_and_preprocess
result = subprocess.run([
    "nnUNetv2_plan_and_preprocess",
    "-d", "1",
    "-c", "3d_lowres",
    "--verify_dataset_integrity",
    "-np", "1"
], env=os.environ, capture_output=False)