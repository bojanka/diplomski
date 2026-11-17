import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np
import os
import csv
from sklearn.decomposition import PCA
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import cKDTree

mask_base = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\HaN-Seg\set_1"
results_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\rezultati"

def load_mesh(mask_path):
    mask = sitk.ReadImage(mask_path)
    spacing = mask.GetSpacing()
    mask_array = sitk.GetArrayFromImage(mask).astype(np.uint8)
    
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

    return normals.GetOutput()

def get_pts(mesh):
    points = mesh.GetPoints()
    n = points.GetNumberOfPoints()
    return np.array([points.GetPoint(i) for i in range(n)])

def surface_area(mesh):
    mass = vtk.vtkMassProperties()
    mass.SetInputData(mesh)
    mass.Update()
    return mass.GetSurfaceArea()

def hausdorff(pts, center_x):
    levo = pts[pts[:, 0] < center_x].copy()
    levo[:, 0] = 2 * center_x - levo[:, 0]
    desno = pts[pts[:, 0] >= center_x]
    if len(levo) > 0 and len(desno) > 0:
        h1 = directed_hausdorff(levo, desno)[0]
        h2 = directed_hausdorff(desno, levo)[0]
        return max(h1, h2)
    return 0

def icp(pts, center_x):
    levo = pts[pts[:, 0] < center_x].copy()
    levo[:, 0] = 2 * center_x - levo[:, 0]
    desno = pts[pts[:, 0] >= center_x]
    if len(levo) > 0 and len(desno) > 0:
        tree = cKDTree(desno)
        dists, _ = tree.query(levo, k=1)
        return dists.mean()
    return 0

def mean_curvature(mesh):
    curvature = vtk.vtkCurvatures()
    curvature.SetInputData(mesh)
    curvature.SetCurvatureTypeToMean()
    curvature.Update()
    scalars = curvature.GetOutput().GetPointData().GetScalars()
    vals = np.array([abs(scalars.GetValue(i)) for i in range(scalars.GetNumberOfTuples())])
    return vals.mean(), vals.std()

def pca_features(pts):
    pca = PCA(n_components=3)
    pca.fit(pts)
    return pca.explained_variance_ratio_

# MAIN
all_results = []

for i in range(1, 43):
    case = f"case_{i:02d}"
    mask_path = os.path.join(mask_base, case, f"{case}_OAR_Bone_Mandible.seg.nrrd")
    
    if not os.path.exists(mask_path):
        print(f"{case} - preskacam")
        continue
    
    print(f"Obrada {case}...")
    
    try:
        mesh = load_mesh(mask_path)
        pts = get_pts(mesh)
        center_x = pts[:, 0].mean()

        area = surface_area(mesh)
        hd = hausdorff(pts, center_x)
        icp_val = icp(pts, center_x)
        curv_mean, curv_std = mean_curvature(mesh)
        evr = pca_features(pts)

        result = {
            'case': case,
            'povrsina_mm2': round(area, 2),
            'hausdorff_mm': round(hd, 3),
            'icp_mm': round(icp_val, 3),
            'zakrivljenost_mean': round(float(curv_mean), 6),
            'zakrivljenost_std': round(float(curv_std), 6),
            'pca_var1': round(float(evr[0]), 4),
            'pca_var2': round(float(evr[1]), 4),
            'pca_var3': round(float(evr[2]), 4),
            'srednja_ravan_x': round(center_x, 3),
            'broj_tacaka': pts.shape[0],
        }

        all_results.append(result)
        print(f"  -> Povrsina: {area:.1f}mm², Hausdorff: {hd:.2f}mm, ICP: {icp_val:.3f}mm, Zakrivljenost: {curv_mean:.4f}")

    except Exception as e:
        print(f"{case} - greska: {e}")

csv_path = os.path.join(results_path, "parametri_pouzdani.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
    writer.writeheader()
    writer.writerows(all_results)

print(f"\nGotovo! Sacuvano u: {csv_path}")
print(f"Obradjeno: {len(all_results)} pacijenata")