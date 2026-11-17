import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

ct_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\HaN-Seg\set_1\case_01\case_01_IMG_CT.nrrd"
mask_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\HaN-Seg\set_1\case_01\case_01_OAR_Bone_Mandible.seg.nrrd"
results_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\rezultati"

ct = sitk.ReadImage(ct_path)
mask = sitk.ReadImage(mask_path)

# Konvertovanje u numpy
ct_array = sitk.GetArrayFromImage(ct)
mask_array = sitk.GetArrayFromImage(mask)

print(f"CT dimenzije: {ct_array.shape}")
print(f"Maska dimenzije: {mask_array.shape}")
print(f"CT vrednosti: min={ct_array.min()}, max={ct_array.max()}")
print(f"Maska unique vrednosti: {np.unique(mask_array)}")

# Nadji slicove gde je mandibula vidljiva
mandible_slices = np.where(mask_array.sum(axis=(1,2)) > 0)[0]
print(f"Mandibula vidljiva na slice-ovima: {mandible_slices[0]} do {mandible_slices[-1]}")

mid_slice = mandible_slices[len(mandible_slices)//2] # Prikazi srednji slice mandibule

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(ct_array[mid_slice], cmap='gray')
axes[0].set_title(f'CT snimak - slice {mid_slice}')

axes[1].imshow(ct_array[mid_slice], cmap='gray')
axes[1].imshow(mask_array[mid_slice], cmap='Reds', alpha=0.5)
axes[1].set_title(f'CT + maska mandibule - slice {mid_slice}')

plt.tight_layout()
# plt.savefig('vizualizacija.png')
plt.savefig(f'{results_path}\\vizualizacija_case_01.png')
plt.show()
print("Saving done")