import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np

mask_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\HaN-Seg\set_1\case_17\case_17_OAR_Bone_Mandible.seg.nrrd"

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

print(f"Prosecno odstupanje: {distances.mean():.3f} mm")
print(f"Maksimalno odstupanje: {distances.max():.3f} mm")
print(f"Minimalno odstupanje: {distances.min():.3f} mm")

vtk_distances = numpy_support.numpy_to_vtk(distances, deep=True)
vtk_distances.SetName("Asymmetry")
mesh.GetPointData().SetScalars(vtk_distances)

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputData(mesh)
mapper.SetScalarModeToUsePointData()
mapper.SetColorModeToMapScalars()
mapper.SetScalarRange(distances.min(), distances.max())

lut = vtk.vtkLookupTable()
lut.SetNumberOfColors(256)
lut.SetHueRange(0.667, 0.0)
lut.Build()
mapper.SetLookupTable(lut)

actor = vtk.vtkActor()
actor.SetMapper(mapper)

scalar_bar = vtk.vtkScalarBarActor()
scalar_bar.SetLookupTable(lut)
scalar_bar.SetTitle("")
scalar_bar.SetNumberOfLabels(6)
scalar_bar.SetPosition(0.88, 0.15)
scalar_bar.SetWidth(0.08)
scalar_bar.SetHeight(0.65)

scalar_bar.GetTitleTextProperty().SetFontSize(18)
scalar_bar.GetTitleTextProperty().SetBold(1)
scalar_bar.GetTitleTextProperty().SetColor(1, 1, 1)
scalar_bar.GetTitleTextProperty().SetFontFamilyToArial()

scalar_bar.GetLabelTextProperty().SetFontSize(14)
scalar_bar.GetLabelTextProperty().SetBold(0)
scalar_bar.GetLabelTextProperty().SetColor(1, 1, 1)
scalar_bar.GetLabelTextProperty().SetFontFamilyToArial()

scalar_bar.SetLabelFormat(" %.2f")
scalar_bar.SetBarRatio(0.3)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.AddActor2D(scalar_bar)
renderer.SetBackground(0.15, 0.15, 0.2)

light1 = vtk.vtkLight()
light1.SetPosition(200, 200, 200)
light1.SetIntensity(0.8)
renderer.AddLight(light1)

light2 = vtk.vtkLight()
light2.SetPosition(-200, 100, -100)
light2.SetIntensity(0.4)
renderer.AddLight(light2)

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)
window.SetSize(800, 600)
window.SetWindowName("Analiza asimetrije")

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)

renderer.ResetCamera()
camera = renderer.GetActiveCamera()
camera.Elevation(0)
camera.Azimuth(0)
renderer.ResetCameraClippingRange()

window.Render()
interactor.Start()