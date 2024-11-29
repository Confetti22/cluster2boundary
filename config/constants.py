# config.py

import numpy as np

config = {
    "image_paths": {
        "img_pth": "/home/confetti/data/mousebrainatlas/T11.ims.h5",
        "mask_pth": "/home/confetti/data/mousebrainatlas/new_t11/registered_atlas.tiff",
    },
    "level": 0,
    "roi_size": np.array([32, 32, 32]),  # (z, y, x) order
    "zoom_factor": 25 / 4,  # atlas_vs/raw_vs

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