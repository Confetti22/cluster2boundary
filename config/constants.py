# config.py

import numpy as np

config = {
    "image_paths": {
        "img_pth": "/home/confetti/mnt/data/processed/t1779/t1779.ims",
        "mask_pth": "/home/confetti/mnt/data/processed/t1779/r32_ims_downsample_561_register/registered_atlas.tiff",
    },
    "level": 0,
    "channel": 2, #channel of interest
    "mip_thickness" :8,
    "roi_size": np.array([8, 256, 256]),  # (z, y, x) order
    "apply_roi_mip": True,  #if True, will apply mip along axis 0(z_dim) of roi
    "zoom_factor": 25 / 1,  # atlas_vs/raw_vs
    "roi_offset" : [11411,4847,4049], #if set, will show this defined roi first 
    "2d_downsample_level":4,
    

    # Plane parameters for ROI and mask
    "roi_plane_parameters": {
        "position": np.array([12, 32, 32]) // 2,
        "normal": (1, 0, 0),
        "thickness": 2,
    },
    "mask_plane_parameters": {
        "position": np.array([12, 32, 32]) // 2,
        "normal": (1, 0, 0),
        "thickness": 1,
    },
    "svc_parameters":{
        "C" : 5,
        "gamma" : 0.05,
        "kernel" : 'rbf',
        "classweight" : None,
        "norm" : True,
    },
}