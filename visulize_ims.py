import napari
from napari_helper.read_ims import Ims_Image
from config.constants import config
import numpy as np


roi_plane_parameters = config["roi_plane_parameters"]

ims_pth = "/home/confetti/mnt/data/processed/t1779/t1779.ims"
ims_vol = Ims_Image(ims_pth, channel=2)
level = 0
z_idx=7802
shape=ims_vol.rois[level][3:]
vol=ims_vol.from_roi(coords=(z_idx,3049,3048,600,128,128),level=level)

vol=np.squeeze(vol)
viewer = napari.Viewer(ndisplay=2)
viewer.add_image(vol)
viewer.add_image(vol,name="roi_plane",depiction='plane',rendering='mip',blending='additive',opacity=0.6,plane=roi_plane_parameters)

napari.run()

