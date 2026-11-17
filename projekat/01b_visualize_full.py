import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ct_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\HaN-Seg\set_1\case_01\case_01_IMG_CT.nrrd"
mask_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\HaN-Seg\set_1\case_01\case_01_OAR_Bone_Mandible.seg.nrrd"
results_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\rezultati"

ct = sitk.ReadImage(ct_path)
mask = sitk.ReadImage(mask_path)
ct_array = sitk.GetArrayFromImage(ct)
mask_array = sitk.GetArrayFromImage(mask)

# Pronadji slajsove sa mandibulom
mandible_slices = np.where(mask_array.sum(axis=(1,2)) > 0)[0]
print(f"Mandibula vidljiva na {len(mandible_slices)} slajsova: {mandible_slices[0]} do {mandible_slices[-1]}")

# Grid prikaz 9 slajsova
indices = np.linspace(0, len(mandible_slices)-1, 9, dtype=int)
selected_slices = mandible_slices[indices]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fig.suptitle('Mandibula - prikaz kroz sve slajsove (case_01)', fontsize=16)

for idx, (ax, slice_idx) in enumerate(zip(axes.flat, selected_slices)):
    ax.imshow(ct_array[slice_idx], cmap='gray')
    ax.imshow(mask_array[slice_idx], cmap='Reds', alpha=0.5)
    ax.set_title(f'Slajs {slice_idx}')
    ax.axis('off')

plt.tight_layout()
plt.savefig(f'{results_path}\\grid_prikaz_case_01.png', dpi=150, bbox_inches='tight')
plt.show()
print("Grid done")

# Animacija
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 6))
fig2.suptitle('Animacija - prolaz kroz mandibulu (case_01)', fontsize=14)

im1 = axes2[0].imshow(ct_array[mandible_slices[0]], cmap='gray', animated=True)
im2 = axes2[1].imshow(ct_array[mandible_slices[0]], cmap='gray', animated=True)
im3 = axes2[1].imshow(mask_array[mandible_slices[0]], cmap='Reds', alpha=0.5, animated=True)

axes2[0].set_title('CT snimak')
axes2[1].set_title('CT + maska mandibule')
axes2[0].axis('off')
axes2[1].axis('off')

def update(frame):
    slice_idx = mandible_slices[frame]
    im1.set_array(ct_array[slice_idx])
    im2.set_array(ct_array[slice_idx])
    im3.set_array(mask_array[slice_idx])
    fig2.suptitle(f'Animacija - slajs {slice_idx} (case_01)', fontsize=14)
    return im1, im2, im3

ani = animation.FuncAnimation(
    fig2, update, frames=len(mandible_slices),
    interval=100, blit=True
)

ani.save(f'{results_path}\\animacija_case_01.gif', writer='pillow', fps=10)
plt.show()
print("Animation done")