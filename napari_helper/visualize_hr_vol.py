import tifffile as tif
from read_ims import Ims_Image
import numpy as np
import os
import napari
from skimage.measure import shannon_entropy
from scipy.ndimage import zoom


# give indexes in hr_vol and get roi in hr_vol 
# compute correspond indexes in mask and get lr_mask
# zoom lr_mask to hr_mask
#add hr_mask and roi_vol to napari



os.environ['DISPLAY'] = ':1'
img_pth="/home/confetti/mnt/data/processed/new_t11/T11.ims.h5"
mask_pth="/home/confetti/mnt/data/processed/new_t11/r0_register/registered_atlas.tiff"


level=0

#roi_size in order of (z,y,x)
roi_size=[64,64,64]
zoom_factor=25/4 #zoom_factor=atlas_vs/raw_vs

ims_vol=Ims_Image(img_pth,channel=3)
lr_mask=tif.imread(mask_pth)
hr_vol_shape=ims_vol.info[level]['data_shape']




viewer = napari.Viewer(ndisplay=3)
viewer.add_image(lr_mask[0:16,0:16,0:16],name='mask')
# viewer.add_image(ims_vol.from_roi(coords=(70,860,860,20,256,256),level=0),name="z_gap1")
# viewer.add_image(ims_vol.from_roi(coords=(145,860,860,20,256,256),level=0),name="z_gap2")
# viewer.add_image(ims_vol.from_roi(coords=(220,860,860,20,256,256),level=0),name="z_gap3")



def get_hr_mask(lr_mask,indexs,roi_size,scale):
    """
    roi is defined via indexs +roi_size
    """
    lr_indexs=[int(idx/scale)for idx in indexs]
    lr_roi_size=[int(roi/scale)for roi in roi_size]
    z,y,x=lr_indexs
    z_size,y_size,x_size=lr_roi_size

    #mask in order (z,y,x)
    lr_mask_roi=lr_mask[z:z+z_size,y:y+y_size,x:x+x_size]
    zoom_factors=[t/s for t,s in zip(roi_size,lr_roi_size)]
    zoomed_mask_roi=zoom(lr_mask_roi,zoom=zoom_factors,order=0)

    return zoomed_mask_roi

def get_aux_c_slice(ims_vol,idx,roi_size):
    """
    center=indexs + roi_size//2
    get z slice from[center_z:center_z+1,:,:]

    """
    center_z=idx[0]+roi_size[0]//2
    z_slice=ims_vol.from_slice(center_z,level=0,index_pos=0)

    #render the roi border at z_slice by draw a 2d cube
    max_value=np.percentile(z_slice,99)
    # z_slice[idx[1]:idx[1]+roi_size[1]       ,       idx[2]]=max_value
    # z_slice[idx[1]+roi_size[1]              ,       idx[2]:idx[2]+roi_size[2]]=max_value
    # z_slice[idx[1]:idx[1]+roi_size[1]       ,       idx[2]+roi_size[2]]=max_value
    # z_slice[idx[1]                          ,       idx[2]:idx[2]+roi_size[2]]=max_value
    z_slice[idx[1]:idx[1]+roi_size[1],idx[2]:idx[2]+roi_size[2]]+=500
    

    return z_slice

    

def entropy_filter(thres=1.2):
    def _filter(img):
        entropy=shannon_entropy(img)
        return entropy>thres
    return _filter


@viewer.mouse_drag_callbacks.append
def get_event(viewer, event):
    print(f"mouse_drag_callback triggered by event:{event.position}")
    pos_on_slice=viewer.layers['aux_slice'].world_to_data(event.position)
    pos_on_roi=viewer.layers['roi'].world_to_data(event.position)
    print(f"viewer_current_step:{viewer.dims.current_step}")
    print(f"viewer_cursor.position:{viewer.cursor.position}")
    print(f"pos on slice:{pos_on_slice}")
    print(f"pos on roi:{pos_on_roi}")

@viewer.mouse_double_click_callbacks.append
def on_second_click_of_double_click(layer, event):
    print('Second click of double_click fectch new roi and generate mask from low resolution')

    new_roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(thres=1.3),roi_size=roi_size,level=0)
    print(f"roi start at {indexs} with shape{roi_size}")

    new_mask=get_hr_mask(lr_mask,indexs,roi_size,zoom_factor)

    # auxiliary z-slice(center at roi) to better known the loaction of roi
    aux_c_slice=get_aux_c_slice(ims_vol,indexs,roi_size)

    mask_classes=np.unique(new_mask)
    print(f"mask id include :{mask_classes}")


    #remove old layers
    if "roi" in viewer.layers:
        viewer.layers.remove("roi")
    
    if "mask" in viewer.layers:
        viewer.layers.remove("mask")

    if "aux_slice" in viewer.layers:
        viewer.layers.remove("aux_slice")
        

    cube_translation_to_plane=indexs.copy()
    cube_translation_to_plane[0]=-roi_size[0]//2
    viewer.add_image(new_roi,translate=cube_translation_to_plane,contrast_limits=(0,np.percentile(new_roi,100)),name="roi")
    viewer.add_labels(new_mask,translate=cube_translation_to_plane,name='mask')

    viewer.add_image(aux_c_slice,name='aux_slice',opacity=0.6)
    


if __name__ == '__main__':
    napari.run()