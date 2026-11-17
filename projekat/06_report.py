import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np

mask_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\HaN-Seg\set_1\case_01\case_01_OAR_Bone_Mandible.seg.nrrd"

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

mesh = normals.GetOutput()
points = mesh.GetPoints()
n_points = points.GetNumberOfPoints()
pts = np.array([points.GetPoint(i) for i in range(n_points)])

center_x = pts[:, 0].mean()
center_z = pts[:, 2].mean()

# Mirrored mesh
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
    abs(distance_filter.EvaluateFunction(pts[i]))
    for i in range(n_points)
])

# Podela na regione po Z osi (gore/dole) i X osi (levo/desno)
z_max = pts[:, 2].max()
z_min = pts[:, 2].min()
z_threshold = z_min + (z_max - z_min) * 0.6  # gornji deo = kondili i ramus

# Regioni
kondili_levo = (pts[:, 2] > z_threshold) & (pts[:, 0] < center_x)
kondili_desno = (pts[:, 2] > z_threshold) & (pts[:, 0] >= center_x)
telo_levo = (pts[:, 2] <= z_threshold) & (pts[:, 0] < center_x)
telo_desno = (pts[:, 2] <= z_threshold) & (pts[:, 0] >= center_x)

print("=" * 50)
print("IZVESTAJ ANALIZE ASIMETRIJE MANDIBULE")
print("=" * 50)
print(f"\nUKUPNA ANALIZA:")
print(f"  Prosecno odstupanje:    {distances.mean():.3f} mm")
print(f"  Maksimalno odstupanje:  {distances.max():.3f} mm")
print(f"  Minimalno odstupanje:   {distances.min():.3f} mm")
print(f"  Std devijacija:         {distances.std():.3f} mm")

print(f"\nPO REGIONIMA:")
print(f"  Kondil/Ramus LEVO:   prosek {distances[kondili_levo].mean():.3f} mm, max {distances[kondili_levo].max():.3f} mm")
print(f"  Kondil/Ramus DESNO:  prosek {distances[kondili_desno].mean():.3f} mm, max {distances[kondili_desno].max():.3f} mm")
print(f"  Telo LEVO:           prosek {distances[telo_levo].mean():.3f} mm, max {distances[telo_levo].max():.3f} mm")
print(f"  Telo DESNO:          prosek {distances[telo_desno].mean():.3f} mm, max {distances[telo_desno].max():.3f} mm")

# Indeks asimetrije
ai = (distances.mean() / (pts[:, 0].max() - pts[:, 0].min())) * 100
print(f"\nINDEKS ASIMETRIJE: {ai:.2f}%")
print("  (0% = savrsena simetrija, vece vrednosti = veca asimetrija)")

# Sacuvaj u fajl
with open(r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\rezultati\izvestaj_case_01.txt", "w") as f:
    f.write("IZVESTAJ ANALIZE ASIMETRIJE MANDIBULE - case_01\n")
    f.write("=" * 50 + "\n")
    f.write(f"Prosecno odstupanje: {distances.mean():.3f} mm\n")
    f.write(f"Maksimalno odstupanje: {distances.max():.3f} mm\n")
    f.write(f"Std devijacija: {distances.std():.3f} mm\n")
    f.write(f"Indeks asimetrije: {ai:.2f}%\n")
    f.write(f"Kondil/Ramus LEVO - prosek: {distances[kondili_levo].mean():.3f} mm\n")
    f.write(f"Kondil/Ramus DESNO - prosek: {distances[kondili_desno].mean():.3f} mm\n")
    f.write(f"Telo LEVO - prosek: {distances[telo_levo].mean():.3f} mm\n")
    f.write(f"Telo DESNO - prosek: {distances[telo_desno].mean():.3f} mm\n")

print(f"\nIzvestaj sacuvan u: izvestaj_case_01.txt")