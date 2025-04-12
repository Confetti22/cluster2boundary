# config.py

import numpy as np

config = {
    "image_paths": {
        "img_pth": "/home/confetti/e5_data/t1779/t1779.ims",
        "mask_pth": "/home/confetti/e5_data/t1779/register_data/r32_ims_downsample_561_register/registered_atlas.tiff",
    },
    "level": 0,
    "channel": 2, # for brain t1779  640 is channel2
    "mip_thickness" :8,
    "roi_size": np.array([1536,64,1536]),  # (z, y, x) order
    "apply_roi_mip": False,  #if True, will apply mip along axis 0(z_dim) of roi
    "zoom_factor": 25 / 1,  # atlas_vs/raw_vs
    "roi_offset" : [7000,3250,4200], #if set (the unit is the highest resolution), will show this defined roi first 
    "2d_downsample_level": 3, #resolution level of the 2d_Coronal slice at sub_viewer
    "save_dir": "/home/confetti/data/t1779/test_data_part_brain",
    "save_mask": True,
    "cnt":5,
    "opacity" :0.28,
    'SHOW_ANNO':  True, #control whether to show region annotation on mask
    'acronym':True,
    # "show_regions": ['fi','LD'], #set 'None' to not use this term
    "show_regions": None,
    "navigation_region":None, 
      #show the 3d-sub-volume wrap the region of the on-side of region
    # "show_regions": None, 

    

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