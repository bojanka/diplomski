import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np
import os
import csv

base_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\HaN-Seg\set_1"
results_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\rezultati"

results = []

for i in range(1, 43):
    case = f"case_{i:02d}"
    mask_path = os.path.join(base_path, case, f"{case}_OAR_Bone_Mandible.seg.nrrd")
    
    if not os.path.exists(mask_path):
        print(f"{case} - maska ne postoji, preskacam")
        continue
    
    print(f"Obrada {case}...")
    
    try:
        mask = sitk.ReadImage(mask_path)
        spacing = mask.GetSpacing()
        mask_array = sitk.GetArrayFromImage(mask).astype(np.uint8)
        
        if mask_array.sum() == 0:
            print(f"{case} - prazna maska, preskacam")
            continue

        vtk_image = vtk.vtkImageData()
        vtk_image.SetDimensions(mask_array.shape[2], mask_array.shape[1], mask_array.shape[0])
        vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])
        vtk_array = numpy_support.numpy_to_vtk(
            mask_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
        vtk_image.GetPointData().SetScalars(vtk_array)

        marching = vtk.vtkMarchingCubes()
        marching.SetInputData(vtk_image)
        marching.SetValue(0, 0.5)
        marching.Update()

        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(marching.GetOutputPort())
        smoother.SetNumberOfIterations(100)
        smoother.SetRelaxationFactor(0.05)
        smoother.Update()

        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(smoother.GetOutputPort())
        normals.ConsistencyOn()
        normals.AutoOrientNormalsOn()
        normals.Update()

        mesh = normals.GetOutput()
        points = mesh.GetPoints()
        n_points = points.GetNumberOfPoints()
        pts = np.array([points.GetPoint(i) for i in range(n_points)])

        center_x = pts[:, 0].mean()

        mirrored_pts = pts.copy()
        mirrored_pts[:, 0] = 2 * center_x - pts[:, 0]

        mirrored_mesh = vtk.vtkPolyData()
        mirrored_points = vtk.vtkPoints()
        for p in mirrored_pts:
            mirrored_points.InsertNextPoint(p)
        mirrored_mesh.SetPoints(mirrored_points)
        mirrored_mesh.SetPolys(mesh.GetPolys())

        distance_filter = vtk.vtkImplicitPolyDataDistance()
        distance_filter.SetInput(mirrored_mesh)

        distances = np.array([
            abs(distance_filter.EvaluateFunction(pts[j]))
            for j in range(n_points)
        ])

        z_max = pts[:, 2].max()
        z_min = pts[:, 2].min()
        z_threshold = z_min + (z_max - z_min) * 0.6

        kondili_levo = (pts[:, 2] > z_threshold) & (pts[:, 0] < center_x)
        kondili_desno = (pts[:, 2] > z_threshold) & (pts[:, 0] >= center_x)
        telo_levo = (pts[:, 2] <= z_threshold) & (pts[:, 0] < center_x)
        telo_desno = (pts[:, 2] <= z_threshold) & (pts[:, 0] >= center_x)

        ai = (distances.mean() / (pts[:, 0].max() - pts[:, 0].min())) * 100

        result = {
            "case": case,
            "mean_mm": round(distances.mean(), 3),
            "max_mm": round(distances.max(), 3),
            "std_mm": round(distances.std(), 3),
            "ai_percent": round(ai, 2),
            "kondil_levo_mean": round(distances[kondili_levo].mean(), 3),
            "kondil_desno_mean": round(distances[kondili_desno].mean(), 3),
            "telo_levo_mean": round(distances[telo_levo].mean(), 3),
            "telo_desno_mean": round(distances[telo_desno].mean(), 3),
        }
        results.append(result)
        print(f"  -> AI: {ai:.2f}%, prosek: {distances.mean():.3f}mm, max: {distances.max():.3f}mm")

    except Exception as e:
        print(f"{case} - greska: {e}")

csv_path = os.path.join(results_path, "rezultati_svi.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nGotovo! Obradjeno {len(results)} pacijenata.")
print(f"Rezultati sacuvani u: {csv_path}")

ais = [r["ai_percent"] for r in results]
means = [r["mean_mm"] for r in results]
print(f"\nSUMARNE STATISTIKE:")
print(f"  Prosecni AI across all cases: {np.mean(ais):.2f}%")
print(f"  Max AI: {max(ais):.2f}% ({results[ais.index(max(ais))]['case']})")
print(f"  Min AI: {min(ais):.2f}% ({results[ais.index(min(ais))]['case']})")
print(f"  Prosecno odstupanje across all: {np.mean(means):.3f}mm")