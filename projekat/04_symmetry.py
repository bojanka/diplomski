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

print(f"Broj tacaka mesh-a: {n_points}")
print(f"Centar mase: {pts.mean(axis=0)}")

# Ravnina simetrije prolazi kroz centar, normala X osa
# Mandibula je simetricna po X osi LD
center = pts.mean(axis=0)
print(f"\nRavnina simetrije:")
print(f"  Prolazi kroz tacku: {center}")
print(f"  Normala: [1, 0, 0] (X osa - levo/desno)")

# Podeli tacke na levu i desnu stranu
left_pts = pts[pts[:, 0] < center[0]]
right_pts = pts[pts[:, 0] >= center[0]]

print(f"\nBroj tacaka leva strana: {len(left_pts)}")
print(f"Broj tacaka desna strana: {len(right_pts)}")

mesh.GetPointData().SetScalars(None)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(mesh)

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0.95, 0.88, 0.75)
actor.GetProperty().SetOpacity(0.8)
actor.GetProperty().SetAmbient(0.3)
actor.GetProperty().SetDiffuse(0.8)

# Ravnina simetrije 
plane_source = vtk.vtkPlaneSource()
plane_source.SetCenter(center[0], center[1], center[2])
plane_source.SetNormal(1, 0, 0)
plane_source.SetPoint1(center[0], center[1] - 60, center[2])
plane_source.SetPoint2(center[0], center[1], center[2] + 60)
plane_source.Update()

plane_mapper = vtk.vtkPolyDataMapper()
plane_mapper.SetInputConnection(plane_source.GetOutputPort())

plane_actor = vtk.vtkActor()
plane_actor.SetMapper(plane_mapper)
plane_actor.GetProperty().SetColor(1.0, 0, 0)
plane_actor.GetProperty().SetOpacity(0.4)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.AddActor(plane_actor)
renderer.SetBackground(0.15, 0.15, 0.2)

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)
window.SetSize(800, 600)
window.SetWindowName("Ravnina simetrije")

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)

renderer.ResetCamera()
camera = renderer.GetActiveCamera()
camera.Elevation(0)
camera.Azimuth(0)
renderer.ResetCameraClippingRange()

window.Render()
interactor.Start()