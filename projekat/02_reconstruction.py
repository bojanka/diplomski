import SimpleITK as sitk
import vtk
from vtk.util import numpy_support
import numpy as np
import imageio

mask_path = r"D:\FTN\FTN4\1. DIPLOMSKI\Baze\HaN-Seg\HaN-Seg\set_1\case_01\case_01_OAR_Bone_Mandible.seg.nrrd"

# Ucitavanje masku
mask = sitk.ReadImage(mask_path)
mask_array = sitk.GetArrayFromImage(mask)

print(f"Maska dimenzije: {mask_array.shape}")
print(f"Broj voksela mandibule: {np.sum(mask_array > 0)}")

# Konvertovanje u VTK format
mask_array_vtk = mask_array.astype(np.uint8)
vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(mask_array_vtk.shape[2],
                         mask_array_vtk.shape[1],
                         mask_array_vtk.shape[0])
vtk_image.SetSpacing(mask.GetSpacing())

vtk_array = numpy_support.numpy_to_vtk(
    mask_array_vtk.ravel(), deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
vtk_image.GetPointData().SetScalars(vtk_array)

# Marching Cubes / Generisanje mesha
marching = vtk.vtkMarchingCubes()
marching.SetInputData(vtk_image)
marching.SetValue(0, 0.5)
marching.Update()

print(f"Broj trouglova u mesh-u: {marching.GetOutput().GetNumberOfCells()}")

mapper = vtk.vtkPolyDataMapper()
mapper.SetInputConnection(marching.GetOutputPort())
mapper.ScalarVisibilityOff()

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(0.95, 0.88, 0.75)

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

window.Render()

# ---------------------------------------------------------
# Snimanje GIF animacije rotacije konacnog modela
# ---------------------------------------------------------
n_frames = 72          # broj frejmova (72 = korak od 5 stepeni)
angle_step = 360.0 / n_frames
duration_per_frame = 0.05   # sekunde po frejmu (~20 fps)

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
    arr = np.flipud(arr)  # VTK cuva sliku odozdo na gore, treba je okrenuti

    frames.append(arr)
    camera.Azimuth(angle_step)

gif_path = r"D:\FTN\FTN4\1. DIPLOMSKI\mandibula_rotacija.gif"
imageio.mimsave(gif_path, frames, duration=duration_per_frame, loop=0)
print(f"GIF animacija sacuvana: {gif_path}")

# ---------------------------------------------------------
# Interaktivni prikaz (posle snimanja GIF-a)
# ---------------------------------------------------------
interactor.Start()