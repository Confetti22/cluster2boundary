import numpy as np
from skimage.measure import shannon_entropy



def get_random_roi_coordinates(shape,roi_size):
    x_idx=np.random.randint(0,shape[0]-roi_size[0]) 
    y_idx=np.random.randint(0,shape[1]-roi_size[1]) 
    z_idx=np.random.randint(0,shape[2]-roi_size[2]) 
    return x_idx,y_idx,z_idx

