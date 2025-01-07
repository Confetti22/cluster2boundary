
#for choose and select dataset
import numpy as np

config = {
    "image_paths": {
        "img_pth": "/home/confetti/mnt/data/processed/t1779/t1779.ims",
    },
    "level": 0,
    "channel": 2,
    "roi_size": np.array([128, 128, 128]),  # (z, y, x) order
    "save_dir": "/home/confetti/mnt/data/processed/t1779/256roi_sg",
    "mask_save_dir":"/home/confetti/mnt/data/processed/t1779/256roi_sg_mask",

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
}