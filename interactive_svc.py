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
from napari.utils.action_manager import action_manager
from napari_helper.napari_view_utilis import MultipleViewerWidget ,add_point_on_plane,\
      remove_layers_with_patterns,toggle_layer_visibility,link_pos_of_plane_layers
from napari.utils.events import Event
from napari_threedee.annotators import PointAnnotator
from config.constants import config



# Load configuration values
img_pth = config["image_paths"]["img_pth"]
mask_pth = config["image_paths"]["mask_pth"]
level = config["level"]
roi_size = config["roi_size"]
zoom_factor = config["zoom_factor"]
roi_plane_parameters = config["roi_plane_parameters"]
mask_plane_parameters = config["mask_plane_parameters"]

# Read IMS image and TIFF mask
ims_vol = Ims_Image(img_pth, channel=3)
lr_mask = tif.imread(mask_pth)
hr_vol_shape = ims_vol.info[level]['data_shape']



# give indexes in hr_vol and get roi in hr_vol 
# compute correspond indexes in mask and get lr_mask
# zoom lr_mask to hr_mask
#add hr_mask and roi_vol to napari


###### prepare data #######
##########################

### whether ims, mask reside on local or cluster, the speed is same


###### init main napari viewer ########
##################################
svc_predictor = None
annotator = None
#main viewer
viewer = napari.Viewer(ndisplay=3)
dummy_data=np.zeros(shape=roi_size,dtype=int)
viewer.add_image(dummy_data,name='roi')

####### init sub viewer #####
dock_widget=MultipleViewerWidget(viewer)
viewer.window.add_dock_widget(dock_widget,name="sub_viewer")

# adjust size of main_window
origin_geo=viewer.window.geometry()
viewer.window.set_geometry(origin_geo[0],origin_geo[1],origin_geo[2],origin_geo[3])

#sub_viewer, can be treated as a viewer
sub_viewer=viewer.window._dock_widgets['sub_viewer'].widget().viewer_model1



#AttributeError: 'ViewerModel' object has no attribute 'window'
# sub_geo=sub_viewer.window.geometry()


# It's a pointLayer in napari
# right click on plane in 3d to change point type(defined by color)
# Once add/remove point, move point, change point color --> 
# --recompute annotation map
# --boundary finding using svm
# --refresh segmentation result to viewer



#allen color_map map many similar adjacent region into one color, not useful in refined brain region seg
# import json
# with open('config/allen_colormap.json','r') as f:
#     allen_cmap=json.load(f)

# allen_cmap[None]=[0,0,0,0]
# allen_cmap = {k: v + [1] for k, v in allen_cmap.items()}
# allen_cmap = {k: [v[0]/255, v[1]/255, v[2]/255, v[3]] for k, v in allen_cmap.items()}


def get_hr_mask(lr_mask,indexs,roi_size,scale):
    """
    roi is defined via indexs +roi_size
    """
    lr_indexs=[int(idx/scale)for idx in indexs]
    # lr_roi_size=[int(roi/scale)for roi in roi_size]
    lr_roi_size = [max(int(roi / scale), 1) for roi in roi_size]
    z,y,x=lr_indexs
    z_size,y_size,x_size=lr_roi_size

    #mask in order (z,y,x)
    lr_mask_roi=lr_mask[z:z+z_size,y:y+y_size,x:x+x_size]
    zoom_factors=[t/s for t,s in zip(roi_size,lr_roi_size)]
    zoomed_mask_roi=zoom(lr_mask_roi,zoom=zoom_factors,order=0)
    zoomed_mask_roi=np.squeeze(zoomed_mask_roi)


    return zoomed_mask_roi

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

    z_idx=indexs[0]+roi_size[0]//2
    hr_slice_shape=ims_vol.rois[level][-2:]
    slice_mask=get_hr_mask(lr_mask,indexs=[z_idx,0,0],roi_size=[1,*hr_slice_shape],scale=zoom_factor)

    sub_viewer=viewer.window._dock_widgets['sub_viewer'].widget().viewer_model1

    #remove out old aux_slice and it's label
    remove_layers_with_patterns(sub_viewer.layers,['aux'])

    slice_name=f'aux_slice{z_idx}'
    sub_viewer.add_image(aux_z_slice,contrast_limits=(0,np.percentile(aux_z_slice,99)+400),name=slice_name)
    sub_viewer.add_labels(slice_mask,name=f'aux_slice_mask{z_idx}',opacity=0.4,)

    #adjust camera in sub_viewer
    sub_viewer.camera.zoom=2
    sub_viewer.camera.center=(0,indexs[1],indexs[2])


def entropy_filter(thres=1.8):
    def _filter(img):
        entropy=shannon_entropy(img)
        return entropy>thres
    return _filter






def adjust_camera_viewer(zomm_factor=6):
    viewer.camera.zoom=zoom_factor



def on_delete():
    print("!!!!!!!!!!!!svc_predictor has been garbage collected.!!!!!!!!!!!!")

# Create a weak reference to monitor the object

def _on_refesh_roi(new_mask,new_roi):

    mask_classes=np.unique(new_mask)
    print(f"mask id include :{mask_classes}")

    remove_layers_with_patterns(viewer.layers,['roi','mask','plane','points'])

    roi_layer=viewer.add_image(new_roi,contrast_limits=(0,np.percentile(new_roi,99)*1.5),name='roi',visible=False)

    #Todo investigate the true cause of runtime error 
    # set visible to False of label_layer to Fasle will cause runtime error
    # but user will need to unvisible the currently unwanted layer by hand
    # mask_layer=viewer.add_labels(new_mask,name='mask',visible=False)
    mask_layer=viewer.add_labels(new_mask,name='mask')


    roi_plane_layer=viewer.add_image(new_roi, name='roi_plane', depiction='plane',rendering='mip',  blending='additive', opacity=0.6, plane=roi_plane_parameters)
    mask_plane_layer=viewer.add_labels(new_mask, name='mask_plane', depiction='plane',blending='additive', opacity=0.6, plane=mask_plane_parameters,)

    points_layer = viewer.add_points(data=[],name='points',size=2,face_color='cornflowerblue',ndim=3)

    link_pos_of_plane_layers([roi_plane_layer,mask_plane_layer])
    viewer.layers.selection=[roi_plane_layer]

    # create the point annotator
    annotator = PointAnnotator(
        viewer=viewer,
        image_layer=roi_plane_layer,
        mask_layer=mask_plane_layer,
        points_layer=points_layer,
        enabled=True,
        config=config
    )

    svc_predictor=SvcWidget(viewer,points_layer,mask_layer,config=config)

    # Create a weak reference to monitor the object
    weakref.finalize(svc_predictor, on_delete)


@viewer.mouse_double_click_callbacks.append
def on_double_click_on_left_viewer(layer, event):
    print(' double_click at left viewer:fectch new roi and generate mask from low resolution and add aux slice')

    new_roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(thres=1.8),roi_size=roi_size,level=0)
    print(f"roi start at {indexs} with shape{roi_size}")

    new_mask=get_hr_mask(lr_mask,indexs,roi_size,zoom_factor)

    # auxiliary z-slice(center at roi) to better known the loaction of roi
    add_data_in_sub_viewer(ims_vol,indexs,roi_size)

    # _on_refesh_roi(new_mask,new_roi)

    mask_classes=np.unique(new_mask)
    print(f"mask id include :{mask_classes}")

    remove_layers_with_patterns(viewer.layers,['roi','mask','plane','points'])

    roi_layer=viewer.add_image(new_roi,contrast_limits=(0,np.percentile(new_roi,99)*1.5),name='roi',visible=False)

    #Todo investigate the true cause of runtime error 
    # set visible to False of label_layer to Fasle will cause runtime error
    # but user will need to unvisible the currently unwanted layer by hand
    # mask_layer=viewer.add_labels(new_mask,name='mask',visible=False)
    mask_layer=viewer.add_labels(new_mask,name='mask')


    roi_plane_layer=viewer.add_image(new_roi, name='roi_plane', depiction='plane',rendering='mip',  blending='additive', opacity=0.6, plane=roi_plane_parameters)
    mask_plane_layer=viewer.add_labels(new_mask, name='mask_plane', depiction='plane',blending='additive', opacity=0.6, plane=mask_plane_parameters,)

    points_layer = viewer.add_points(data=[],name='points',size=2,face_color='cornflowerblue',ndim=3)

    link_pos_of_plane_layers([roi_plane_layer,mask_plane_layer])
    viewer.layers.selection=[roi_plane_layer]

    # create the point annotator
    annotator = PointAnnotator(
        viewer=viewer,
        image_layer=roi_plane_layer,
        mask_layer=mask_plane_layer,
        points_layer=points_layer,
        enabled=True,
        config=config
    )

    svc_predictor=SvcWidget(viewer,points_layer,mask_layer,config=config)

    # Create a weak reference to monitor the object
    weakref.finalize(svc_predictor, on_delete)



 

@sub_viewer.mouse_double_click_callbacks.append
def on_double_click_at_sub_viewer(_module,event):
    print(f'double click at sub_viewer: type of _moudule')

    #cursor coords(3d) -> data coords(2d) + slice_idx -->roi_center coords -->roi_offset+roi_size--->roi
    mouse_pos=sub_viewer.cursor.position
    
    if any(key.name.startswith('aux') for key in sub_viewer.layers): 
        layer_name=sub_viewer.layers[0].name
        slice_idx = int(layer_name.split('aux_slice')[-1])

        data_coords=sub_viewer.layers[f'aux_slice{slice_idx}'].world_to_data(mouse_pos)
        data_coords=np.round(data_coords).astype(int)

        roi_center_idx=np.insert(data_coords,0,slice_idx)
        roi_offset=roi_center_idx - roi_size//2
        roi_offset = np.maximum(roi_offset, 0)
        print(f"roi_offset:{roi_offset},roi_size:{roi_size}")

        roi=ims_vol.from_roi(coords=[*roi_offset, *roi_size],level=level)
        roi=roi.reshape(*roi_size)

        mask=get_hr_mask(lr_mask,roi_offset,roi_size,zoom_factor)

        _on_refesh_roi(mask,roi)
        adjust_camera_viewer()


@sub_viewer.bind_key('v')
def toggle_mask_sub_viewer(_module):
    print(f'press \'v\' at  {_module}')
    toggle_layer_visibility(layers=sub_viewer.layers,name_patterns=['mask'])

def toggle_mask(viewer_model: napari.components.viewer_model.ViewerModel):
    print(f'press \'v\' at  {viewer_model}')
    toggle_layer_visibility(layers=viewer_model.layers,name_patterns=['mask_plane','mask'])
    # toggle_layer_visibility(layers=sub_viewer.layers,name_patterns=['mask'])

action_manager.register_action(
    name='napari:toggle_mask',
    command=toggle_mask,
    description="toggle mask",
    keymapprovider=ViewerModel,
)

action_manager.bind_shortcut('napari:toggle_mask', 'v')






# @viewer.mouse_drag_callbacks.append
# def get_event(viewer, event):
#     print(event)
#     #first make sure if event.postion equals viewer.cursor.position: yes, they equals

#     print(f"position on event:{event.position}")
#     print(f"position on 3d roi :{viewer.layers['roi'].world_to_data(event.position)}")
#     print(f"position on 2d roi :{viewer.layers['roi_plane'].world_to_data(event.position)}")
#     print(f"position on 2d mask :{viewer.layers['mask_plane'].world_to_data(event.position)}")



# @viewer.mouse_drag_callbacks.append
# def on_mouse_alt_click_add_point_on_plane(
#         viewer: napari.viewer.Viewer,
#         event: Event,
#         points_layer: napari.layers.Points = None,
#         image_layer: napari.layers.Image = None,
#         replace_selected: bool = False,
# ):
#     # Early exit if not alt-clicked
#     if 'Alt' not in event.modifiers:
#         return

#     add_point_on_plane(
#         viewer=viewer,
#         points_layer=points_layer,
#         image_layer=image_layer,
#         replace_selected=replace_selected
#     )






if __name__ == '__main__':
    napari.run()