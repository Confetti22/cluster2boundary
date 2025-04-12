from napari_helper.read_ims import Ims_Image
from skimage.measure import shannon_entropy
import tifffile as tif
import numpy as np
from scipy.ndimage import zoom
import os

def entropy_filter(l_thres=1.4, h_thres=100):
    def _filter(img):
        entropy=shannon_entropy(img)
        if (entropy>= l_thres) and (entropy <= h_thres):
            print(f"entrop of the roi is {entropy}")
            return True
        else:
            return False
    return _filter

def get_mask_from_different_scale(source_mask,target_indexs,target_roi_size,scale):
    """
    roi is defined via indexs +roi_size
    """
    source_indexs=[int(idx/scale)for idx in target_indexs]
    # lr_roi_size=[int(roi/scale)for roi in roi_size]
    source_roi_size = [max(int(roi / scale), 1) for roi in target_roi_size]
    z,y,x=source_indexs
    z_size,y_size,x_size=source_roi_size

    #mask in order (z,y,x)
    lr_mask_roi=source_mask[z:z+z_size,y:y+y_size,x:x+x_size]
    zoom_factors=[t/s for t,s in zip(target_roi_size,source_roi_size)]
    zoomed_mask_roi=zoom(lr_mask_roi,zoom=zoom_factors,order=0)
    zoomed_mask_roi=np.squeeze(zoomed_mask_roi)

    return zoomed_mask_roi

MASK = False
save_dir="/home/confetti/data/t1779/4096_roi_sg"
os.makedirs(save_dir,exist_ok=True)

if MASK:
    mask_save_dir="/home/confetti/mnt/data/processed/t1779/256roi_sg_mask"
    os.makedirs(mask_save_dir,exist_ok=True)
    mask_path = "/home/confetti/mnt/data/processed/t1779/r32_ims_downsample_561_register/registered_atlas.tiff"
    lr_mask = tif.imread(mask_path)



image_path = "/home/confetti/mnt/data/processed/t1779/t1779.ims"
level = 0
channel = 2
roi_size =(64,64,64)
zoom_factor = 25/1
amount = 4096 
cnt = 1
sample_range = [[4000,7000],[0,5000],[0,7000]]

ims_vol = Ims_Image(image_path, channel=channel)
vol_shape = ims_vol.info[level]['data_shape']

while cnt < amount:

    roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(l_thres=4),roi_size=roi_size,level=0,skip_gap = True,sample_range=sample_range)
    file_name = f"{save_dir}/{cnt:04d}.tif"
    tif.imwrite(file_name,roi)
    print(f"{file_name} has been saved ")
    cnt = cnt +1
    if MASK:
        mask=get_mask_from_different_scale(lr_mask,indexs,roi_size,zoom_factor)
        mask_name = f"{mask_save_dir}/{cnt:04d}.tif"
        tif.imwrite(mask_name,mask)



