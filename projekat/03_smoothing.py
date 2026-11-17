import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np
import imageio

mask_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\HaN-Seg\set_1\case_01\case_01_OAR_Bone_Mandible.seg.nrrd"

mask = sitk.ReadImage(mask_path)
spacing = mask.GetSpacing()
mask_array = sitk.GetArrayFromImage(mask).astype(np.uint8)

# VTK image sa ispravnim spacingom
vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(mask_array.shape[2], mask_array.shape[1], mask_array.shape[0])
vtk_image.SetSpacing(spacing[0], spacing[1], spacing[2])

vtk_array = numpy_support.numpy_to_vtk(
    mask_array.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
vtk_image.GetPointData().SetScalars(vtk_array)

# Marching Cubes
marching = vtk.vtkMarchingCubes()
marching.SetInputData(vtk_image)
marching.SetValue(0, 0.5)
marching.Update()

# Smoothing
smoother = vtk.vtkSmoothPolyDataFilter()
smoother.SetInputConnection(marching.GetOutputPort())
smoother.SetNumberOfIterations(100)
smoother.SetRelaxationFactor(0.05)
smoother.Update()

# Normale
normals = vtk.vtkPolyDataNormals()
normals.SetInputConnection(smoother.GetOutputPort())
normals.ConsistencyOn()
normals.AutoOrientNormalsOn()
normals.Update()
normals.GetOutput().GetPointData().SetScalars(None)

print(f"Broj trouglova: {normals.GetOutput().GetNumberOfCells()}")

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(normals.GetOutputPort())

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0.95, 0.88, 0.75) 
actor.GetProperty().SetSpecular(0.2)
actor.GetProperty().SetSpecularPower(15)
actor.GetProperty().SetAmbient(0.3)
actor.GetProperty().SetDiffuse(0.8)

renderer = vtk.vtkRenderer()
renderer.AddActor(actor)
renderer.SetBackground(0.15, 0.15, 0.2)

window = vtk.vtkRenderWindow()
window.AddRenderer(renderer)
window.SetSize(800, 600)
window.SetWindowName("3D Mandibula")

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(window)

renderer.ResetCamera()
camera = renderer.GetActiveCamera()
camera.SetPosition(0, -1, 0)      
camera.SetFocalPoint(0, 0, 0)
camera.SetViewUp(0, 0, 1)         
renderer.ResetCamera()            
camera.Azimuth(0)
camera.Elevation(0)
renderer.ResetCameraClippingRange()

window.Render()

# ---------------------------------------------------------
# Snimanje GIF animacije rotacije konacnog modela
# ---------------------------------------------------------
n_frames = 72
angle_step = 360.0 / n_frames
duration_per_frame = 0.05

frames = []

w2if = vtk.vtkWindowToImageFilter()
w2if.SetInput(window)
w2if.SetInputBufferTypeToRGB()
w2if.ReadFrontBufferOff()

for i in range(n_frames):
    window.Render()
    w2if.Modified()
    w2if.Update()

    vtk_img = w2if.GetOutput()
    width, height, _ = vtk_img.GetDimensions()
    vtk_scalars = vtk_img.GetPointData().GetScalars()
    components = vtk_scalars.GetNumberOfComponents()

    arr = numpy_support.vtk_to_numpy(vtk_scalars).reshape(height, width, components)
    arr = np.flipud(arr)

    frames.append(arr)
    camera.Azimuth(angle_step)

gif_path = r"D:\FTN\FTN4\1. DIPLOMSKI\mandibulaSmooth_rotacija.gif"
imageio.mimsave(gif_path, frames, duration=duration_per_frame, loop=0)
print(f"GIF animacija sacuvana: {gif_path}")

# ---------------------------------------------------------
# Interaktivni prikaz (posle snimanja GIF-a)
# ---------------------------------------------------------
interactor.Start()