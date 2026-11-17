import os
import shutil
import json
import SimpleITK as sitk
import numpy as np

# Putanje
raw_path = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\data\raw"
masks_path = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\data\masks"
nnunet_path = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\data\nnunet"

# nnU-Net struktura
dataset_name = "Dataset001_Mandible"
dataset_path = os.path.join(nnunet_path, dataset_name)

imagesTr = os.path.join(dataset_path, "imagesTr")
labelsTr = os.path.join(dataset_path, "labelsTr")
imagesTs = os.path.join(dataset_path, "imagesTs")

os.makedirs(imagesTr, exist_ok=True)
os.makedirs(labelsTr, exist_ok=True)
os.makedirs(imagesTs, exist_ok=True)

# Podeli dataset - 35 za trening, 7 za test
all_cases = sorted([f.replace("_IMG_CT.nrrd", "") 
                    for f in os.listdir(raw_path) 
                    if f.endswith("_IMG_CT.nrrd")])

train_cases = all_cases[:35]
test_cases = all_cases[35:]

print(f"Trening: {len(train_cases)} slucajeva")
print(f"Test: {len(test_cases)} slucajeva")

# Kopiraj trening podatke
for case in train_cases:
    ct_src = os.path.join(raw_path, f"{case}_IMG_CT.nrrd")
    mask_src = os.path.join(masks_path, f"{case}_OAR_Bone_Mandible.seg.nrrd")
    
    # nnU-Net format - CT mora imati _0000 sufiks
    ct_dst = os.path.join(imagesTr, f"{case}_0000.nrrd")
    mask_dst = os.path.join(labelsTr, f"{case}.nrrd")
    
    # Konvertuj masku u binary int (0 i 1)
    mask = sitk.ReadImage(mask_src)
    mask_array = sitk.GetArrayFromImage(mask)
    mask_binary = (mask_array > 0).astype(np.uint8)
    mask_out = sitk.GetImageFromArray(mask_binary)
    mask_out.CopyInformation(mask)
    
    shutil.copy(ct_src, ct_dst)
    sitk.WriteImage(mask_out, mask_dst)
    print(f"  Kopirano: {case}")

# Kopiraj test podatke
for case in test_cases:
    ct_src = os.path.join(raw_path, f"{case}_IMG_CT.nrrd")
    ct_dst = os.path.join(imagesTs, f"{case}_0000.nrrd")
    shutil.copy(ct_src, ct_dst)
    print(f"  Test: {case}")

# Napravi dataset.json
dataset_json = {
    "channel_names": {"0": "CT"},
    "labels": {"background": 0, "mandible": 1},
    "numTraining": len(train_cases),
    "file_ending": ".nrrd"
}

with open(os.path.join(dataset_path, "dataset.json"), "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"\nGotovo! Dataset pripremljen u: {dataset_path}")
print(f"dataset.json kreiran.")