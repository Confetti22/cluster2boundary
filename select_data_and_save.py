import os
import weakref
import sys
sys.path.insert(0,f'{os.getcwd()}/napari_helper')


import tifffile as tif
import numpy as np
from skimage.measure import shannon_entropy
from scipy.ndimage import zoom
import napari


from napari_helper.read_ims import Ims_Image
from napari_helper._svc_widget import SvcWidget
from napari.components.viewer_model import ViewerModel
from napari_helper.napari_view_utilis import MultipleViewerWidget ,remove_layers_with_patterns
from napari.utils.events import Event
from config.select_constants import config

def get_aux_c_slice(ims_vol,idx,roi_size):
    """
    center=indexs + roi_size//2
    get z slice from[z_idx:z_idx+1,:,:]

    """
    z_idx=idx[0]+roi_size[0]//2
    z_slice=ims_vol.from_slice(z_idx,level=0,index_pos=0)

    #render the roi border at z_slice by draw a 2d cube
    max_value=np.percentile(z_slice,99)

    roi_mean=np.mean(z_slice[idx[1]:idx[1]+roi_size[1],idx[2]:idx[2]+roi_size[2]])
    z_slice[idx[1]:idx[1]+roi_size[1],idx[2]:idx[2]+roi_size[2]]+=int(roi_mean*0.5)

    return z_slice


def add_data_in_sub_viewer(ims_vol,indexs,roi_size):
    aux_z_slice=get_aux_c_slice(ims_vol,indexs,roi_size) 
    downsampled_aux_z = zoom(aux_z_slice , 1/4, order=1)

    z_idx=indexs[0]+roi_size[0]//2

    sub_viewer=viewer.window._dock_widgets['sub_viewer'].widget().viewer_model1

    #remove out old aux_slice and it's label
    remove_layers_with_patterns(sub_viewer.layers,['aux'])

    slice_name=f'aux_slice{z_idx}'
    sub_viewer.add_image(downsampled_aux_z,contrast_limits=(0,np.percentile(downsampled_aux_z,99)+400),name=slice_name)

    #adjust camera in sub_viewer
    sub_viewer.camera.zoom=2
    sub_viewer.camera.center=(0,indexs[1],indexs[2])


def entropy_filter(l_thres=1.4, h_thres=100):
    def _filter(img):
        entropy=shannon_entropy(img)
        if (entropy>= l_thres) and (entropy <= h_thres):
            print(f"entrop of the roi is {entropy}")
            return True
        else:
            return False
    return _filter

def adjust_camera_viewer(zoom_factor=6):
    viewer.camera.zoom=zoom_factor


def on_delete():
    print("!!!!!!!!!!!!svc_predictor has been garbage collected.!!!!!!!!!!!!")

# Create a weak reference to monitor the object
def _on_refresh_roi2(new_roi):
    viewer.layers['roi'].data=new_roi
    viewer.layers['roi'].contrast_limits=(0,np.percentile(new_roi,99)*1.5)





###### prepare data #######
##########################

# Load configuration values
img_pth = config["image_paths"]["img_pth"]
level = config["level"]
channel = config["channel"]
roi_size = config["roi_size"]
save_dir=config['save_dir']

# Read IMS image 
ims_vol = Ims_Image(img_pth, channel=channel)
vol_shape = ims_vol.info[level]['data_shape']

#saved roi number
cnt=11

###### init main napari viewer ########
##################################
#main viewer
viewer = napari.Viewer(ndisplay=3)
####### init sub viewer #####


# dock_widget=MultipleViewerWidget(viewer)
# viewer.window.add_dock_widget(dock_widget,name="sub_viewer")

# adjust size of main_window
# origin_geo=viewer.window.geometry()
# viewer.window.set_geometry(origin_geo[0],origin_geo[1],origin_geo[2],origin_geo[3])

#sub_viewer, can be treated as a viewer
# sub_viewer=viewer.window._dock_widgets['sub_viewer'].widget().viewer_model1

roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(l_thres=1.4),roi_size=roi_size,level=0)
print(f"roi start at {indexs} with shape{roi_size}")

# auxiliary z-slice(center at roi) to better known the loaction of roi
# add_data_in_sub_viewer(ims_vol,indexs,roi_size)


roi_layer=viewer.add_image(roi,contrast_limits=(0,np.percentile(roi,99)*1.5),name='roi',visible=True)




@viewer.mouse_double_click_callbacks.append
def on_double_click_on_left_viewer(layer, event):
    print(' double_click at left viewer:fectch new roi and generate aux slice')

    new_roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(l_thres=1.4),roi_size=roi_size,level=level)
    print(f"roi start at {indexs} with shape{roi_size}")

    # auxiliary z-slice(center at roi) to better known the loaction of roi
    # add_data_in_sub_viewer(ims_vol,indexs,roi_size)

    #refresh new roi in viewer 
    _on_refresh_roi2(new_roi)


 

# @sub_viewer.mouse_double_click_callbacks.append
# def on_double_click_at_sub_viewer(_module,event):
#     print(f'double click at sub_viewer: type of _moudule')

#     #cursor coords(3d) -> data coords(2d) + slice_idx -->roi_center coords -->roi_offset+roi_size--->roi
#     mouse_pos=sub_viewer.cursor.position
    
#     if any(key.name.startswith('aux') for key in sub_viewer.layers): 
#         layer_name=sub_viewer.layers[0].name
#         slice_idx = int(layer_name.split('aux_slice')[-1])

#         data_coords=sub_viewer.layers[f'aux_slice{slice_idx}'].world_to_data(mouse_pos)
#         data_coords=data_coords*4
#         data_coords=np.round(data_coords).astype(int)

#         roi_center_idx=np.insert(data_coords,0,slice_idx)
#         roi_offset=roi_center_idx - roi_size//2
#         roi_offset = np.maximum(roi_offset, 0)
#         print(f"roi_offset:{roi_offset},roi_size:{roi_size}")

#         roi=ims_vol.from_roi(coords=[*roi_offset, *roi_size],level=level)
#         roi=roi.reshape(*roi_size)

#         _on_refresh_roi2(roi)
#         adjust_camera_viewer()


@viewer.bind_key('s')
def save_roi(_module):
    global cnt
    print(f'press \'s\' at  {_module}')
    roi=viewer.layers['roi'].data

    file_name = f'{save_dir}/{cnt:04d}.tif'
    print(f"{file_name} has been saved ")
    tif.imwrite(file_name,roi)
    cnt = cnt +1

    new_roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(l_thres=1.4),roi_size=roi_size,level=level)
    print(f"roi start at {indexs} with shape{roi_size}")
    _on_refresh_roi2(new_roi)




if __name__ == '__main__':
    napari.run()