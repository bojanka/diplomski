import sys
sys.path.insert(0, r"D:\Python_packages")

import os
import SimpleITK as sitk
import numpy as np

predictions_path = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\results\predictions"
ground_truth_path = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\data\masks"
results_path = r"D:\FTN\FTN4\1. DIPLOMSKI\projekat2\results"

test_cases = [f"case_{i:02d}" for i in range(36, 43)]

def dice_coefficient(pred, gt):
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    intersection = np.sum(pred_bin * gt_bin)
    return 2 * intersection / (np.sum(pred_bin) + np.sum(gt_bin) + 1e-8)

print("=" * 50)
print("EVALUACIJA SEGMENTACIJE")
print("=" * 50)

dice_scores = []

for case in test_cases:
    pred_path = os.path.join(predictions_path, f"{case}.nrrd")
    gt_path = os.path.join(ground_truth_path, f"{case}_OAR_Bone_Mandible.seg.nrrd")
    
    if not os.path.exists(pred_path):
        print(f"{case} - predikcija ne postoji, preskacam")
        continue
    
    pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
    gt = sitk.GetArrayFromImage(sitk.ReadImage(gt_path))
    
    dice = dice_coefficient(pred, gt)
    dice_scores.append(dice)
    print(f"{case}: Dice = {dice:.4f}")

print("=" * 50)
print(f"Prosecni Dice: {np.mean(dice_scores):.4f}")
print(f"Min Dice: {np.min(dice_scores):.4f}")
print(f"Max Dice: {np.max(dice_scores):.4f}")

with open(os.path.join(results_path, "evaluacija.txt"), "w") as f:
    f.write("EVALUACIJA SEGMENTACIJE\n")
    f.write("=" * 50 + "\n")
    for i, case in enumerate(test_cases[:len(dice_scores)]):
        f.write(f"{case}: Dice = {dice_scores[i]:.4f}\n")
    f.write(f"\nProsecni Dice: {np.mean(dice_scores):.4f}\n")
    f.write(f"Min Dice: {np.min(dice_scores):.4f}\n")
    f.write(f"Max Dice: {np.max(dice_scores):.4f}\n")

print(f"\nRezultati sacuvani u: evaluacija.txt")