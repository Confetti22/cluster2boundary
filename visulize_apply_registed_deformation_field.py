import numpy as np
import tifffile as tiff
from scipy.ndimage import map_coordinates
import napari

dir_path ="/home/confetti/mnt/data/processed/t1779/r32_ims_downsample_561_register"
# Load the volume and deformation fields
volume = tiff.imread(f"{dir_path}/downsampled.tiff")  # Replace with your volume file path
deformation_field_0 = tiff.imread(f"{dir_path}/deformation_field_0.tiff")
deformation_field_1 = tiff.imread(f"{dir_path}/deformation_field_1.tiff")
deformation_field_2 = tiff.imread(f"{dir_path}/deformation_field_2.tiff")

# Ensure the deformation fields are stacked into a single array
deformation_field = np.stack([deformation_field_2, deformation_field_1, deformation_field_0], axis=-1)  # Z, Y, X ordering
# Transpose deformation_field to match the shape of [z, y, x]
deformation_field = np.transpose(deformation_field, axes=(3, 0, 1, 2))  # Move the last axis to the first position


# Create a grid of coordinates
z, y, x = np.meshgrid(
    np.arange(volume.shape[0]),
    np.arange(volume.shape[1]),
    np.arange(volume.shape[2]),
    indexing="ij"
)

# Apply deformation field to the coordinates
coords = np.array([z, y, x]) + deformation_field
coords = np.clip(coords, 0, np.array(volume.shape)[:, None, None, None] - 1)  # Clip to volume bounds

# Interpolate the volume at the deformed coordinates
deformed_volume = map_coordinates(volume, coords, order=1, mode='nearest')

# Visualize using Napari
viewer = napari.Viewer()
viewer.add_image(volume, name="Original Volume")
viewer.add_image(deformed_volume, name="Deformed Volume")

napari.run()