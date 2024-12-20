import os
import weakref
import sys
sys.path.insert(0,f'{os.getcwd()}/napari_helper')


import tifffile as tif
import numpy as np
import time
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

def get_roi_and_mask_mip_defined_indexes(roi_size,roi_offset,level,zoom_factor,roi_mip_flag=False): 
    roi=ims_vol.from_roi(coords=[*roi_offset, *roi_size],level=level)
    print(f"roi start at {roi_offset} with shape{roi_size}")
    roi=roi.reshape(*np.squeeze(roi_size))
    # mask=get_mask_from_different_scale(lr_mask,roi_offset,roi_size,zoom_factor)
    roi = mip(roi,apply_roi_mip)
    print(f"apply mip is {apply_roi_mip}, after mip, roi_size is {roi.shape}")

    target_roi_size = roi.shape
    if len(roi.shape) == 2:
        target_roi_size = np.insert(target_roi_size,0,1)

    mask=get_mask_from_different_scale(lr_mask,roi_offset,target_roi_size,zoom_factor)
    return roi, mask

def get_roi_and_mask_indexs_mip(roi_size, level, zoom_factor,roi_mip_flag =False ):

    roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(thres=1.8),roi_size=roi_size,level=level)
    roi = mip(roi,roi_mip_flag)

    print(f"roi start at {indexs} with shape{roi_size}")
    print(f"apply mip is {roi_mip_flag}, after mip, roi_size is {roi.shape}")

    target_roi_size = roi.shape
    if len(roi.shape) == 2:
        target_roi_size = np.insert(target_roi_size,0,1)

    mask=get_mask_from_different_scale(lr_mask,indexs,target_roi_size,zoom_factor)
    return roi, mask, indexs

def add_data_in_sub_viewer(ims_vol,hr_indexs,hr_roi_size):

    #get downsampled_idx and roi_size in 2d cornoral slice from hr_indexs and hr_roi_size
    #then get the cornoral slice as aux_z_slice
    down_factor = 2**two_dim_downsample_level
    downsampled_idx = [ int( idx //down_factor) for idx in hr_indexs]
    downsampled_roi_size = [ max(int(edge//down_factor) ,1)for edge in hr_roi_size]
    downsampled_z_idx = downsampled_idx[0] + downsampled_roi_size[0]//2
    
    start = time.time()
    aux_z_slice=ims_vol.from_slice(downsampled_z_idx,level=two_dim_downsample_level,index_pos=0,mip_thick=mip_thickness//(2**two_dim_downsample_level))
    print(f"load 2d slice time {time.time()-start}")

    start = time.time()
    #render the roi border at z_slice by draw a 2d cube
    idx_ =downsampled_idx
    roi_ = downsampled_roi_size
    roi_polygon = np.array( 
        [
            [idx_[1]           , idx_[2]],
            [idx_[1] + roi_[1] , idx_[2]],
            [idx_[1] + roi_[1] , idx_[2] + roi_[2]],
            [idx_[1]           , idx_[2] + roi_[2]],
        ])

    ori_z_idx=hr_indexs[0]+hr_roi_size[0]//2
    hr_slice_shape=ims_vol.rois[level + two_dim_downsample_level][-2:]

    slice_mask=get_mask_from_different_scale(lr_mask,target_indexs=[downsampled_z_idx,0,0],target_roi_size=[1,*hr_slice_shape],scale=zoom_factor / (2**two_dim_downsample_level))

    sub_viewer=viewer.window._dock_widgets['sub_viewer'].widget().viewer_model1

    #remove out old aux_slice and it's label and polygon
    remove_layers_with_patterns(sub_viewer.layers,['aux','polygon'])


    #need to adjust contrast_limit, and using max as upper_bound is almost black, then choose mean
    sub_viewer.add_image(aux_z_slice,name=f'aux_slice{ori_z_idx}',contrast_limits=(0,np.percentile(aux_z_slice,99)+600))
    sub_viewer.add_labels(slice_mask,name=f'aux_slice_mask{ori_z_idx}',opacity=0.4,)
    sub_viewer.add_shapes(roi_polygon,name='polygon',edge_width=1,edge_color='cyan',opacity =0.27)

    #adjust camera in sub_viewer
    sub_viewer.camera.zoom=2
    sub_viewer.camera.center=(0,downsampled_idx[1],downsampled_idx[2])
    print(f"total time {time.time() - start}")


def entropy_filter(thres=1.8):
    def _filter(img):
        entropy=shannon_entropy(img)
        return entropy>thres
    return _filter

def adjust_camera_viewer(zoom_factor=2):
    viewer.camera.zoom=zoom_factor


def on_delete():
    print("!!!!!!!!!!!!svc_predictor has been garbage collected.!!!!!!!!!!!!")

# Create a weak reference to monitor the object
def _on_refresh_roi2(new_mask,new_roi):
    viewer.layers['roi'].data=new_roi
    viewer.layers['roi'].contrast_limits=(0,np.percentile(new_roi,99)*1.5)
    viewer.layers['mask'].data=new_mask

    viewer.layers['points'].data=[]



def _on_refesh_roi(new_mask,new_roi):

    mask_classes=np.unique(new_mask)
    print(f"mask id include :{mask_classes}")

    remove_layers_with_patterns(viewer.layers,['roi','mask','plane','points'])

    roi_layer=viewer.add_image(new_roi,contrast_limits=(0,np.percentile(new_roi,99)*1.5),name='roi',visible=False)

    #Todo investigate the true cause of runtime error 
    # set visible to False of label_layer to Fasle will cause runtime error
    # but user will need to unvisible the currently unwanted layer by hand
    # mask_layer=viewer.add_labels(new_mask,name='mask',visible=False)
    mask_layer=viewer.add_labels(new_mask,name='mask',visible=False)


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

def mip(roi, mip_flag = False):
    if mip_flag:
        roi = np.max(roi, axis=0)
    else:
        roi = roi
    roi = np.squeeze(roi)
    return roi


# Load configuration values
img_pth = config["image_paths"]["img_pth"]
mask_pth = config["image_paths"]["mask_pth"]
level = config["level"]
channel = config['channel']
roi_size = config["roi_size"]
zoom_factor = config["zoom_factor"]
apply_roi_mip = config["apply_roi_mip"]
two_dim_downsample_level = config['2d_downsample_level']
roi_plane_parameters = config["roi_plane_parameters"]
mask_plane_parameters = config["mask_plane_parameters"]
mip_thickness = config['mip_thickness']
roi_offset = config.get('roi_offset')
# Read IMS image and TIFF mask
ims_vol = Ims_Image(img_pth, channel=channel)
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
viewer = napari.Viewer(ndisplay=2)
####### init sub viewer #####

dock_widget=MultipleViewerWidget(viewer)
viewer.window.add_dock_widget(dock_widget,name="sub_viewer")

# adjust size of main_window
origin_geo=viewer.window.geometry()
viewer.window.set_geometry(origin_geo[0],origin_geo[1],origin_geo[2],origin_geo[3])

#sub_viewer, can be treated as a viewer
sub_viewer=viewer.window._dock_widgets['sub_viewer'].widget().viewer_model1



if roi_offset:
    #offset is defined
    roi, mask = get_roi_and_mask_mip_defined_indexes(roi_size,roi_offset,level,zoom_factor,apply_roi_mip)
    add_data_in_sub_viewer(ims_vol,roi_offset,roi_size)
else :
    # if not defined, generate randomly
    roi, mask, indexs = get_roi_and_mask_indexs_mip(roi_size, level, zoom_factor,roi_mip_flag = apply_roi_mip)
    add_data_in_sub_viewer(ims_vol,indexs,roi_size)




mask_classes=np.unique(mask)
print(f"mask id include :{mask_classes}")

roi_layer=viewer.add_image(roi,contrast_limits=(0,np.percentile(roi,99)*1.5),name='roi',visible=True)

#Todo investigate the true cause of runtime error 
# set visible to False of label_layer to Fasle will cause runtime error
# but user will need to unvisible the currently unwanted layer by hand
# mask_layer=viewer.add_labels(new_mask,name='mask',visible=False)
mask_layer=viewer.add_labels(mask,name='mask',visible=False)

points_layer = viewer.add_points(data=[],name='points',size=2,face_color='cornflowerblue',ndim=2)


# create the point annotator
annotator = PointAnnotator(
    viewer=viewer,
    image_layer=roi_layer,
    mask_layer=mask_layer,
    points_layer=points_layer,
    enabled=True,
    config=config
)

svc_predictor=SvcWidget(viewer,sub_viewer,points_layer,mask_layer,config=config)

# Create a weak reference to monitor the object
weakref.finalize(svc_predictor, on_delete)


# It's a pointLayer in napari
# right click on plane in 3d to change point type(defined by color)
# Once add/remove point, move point, change point color --> 
# --recompute annotation map
# --boundary finding using svm
# --refresh segmentation result to viewer




@viewer.mouse_double_click_callbacks.append
def on_double_click_on_left_viewer(layer, event):
    print(' double_click at left viewer:fectch new roi and generate mask from low resolution and add aux slice')

    # new_roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(thres=1.8),roi_size=roi_size,level=0)
    # new_mask=get_mask_from_different_scale(lr_mask,indexs,roi_size,zoom_factor)
    new_roi, new_mask, indexs = get_roi_and_mask_indexs_mip(roi_size, level, zoom_factor,roi_mip_flag = apply_roi_mip)

    # auxiliary z-slice(center at roi) to better known the loaction of roi
    add_data_in_sub_viewer(ims_vol,indexs,roi_size)

    mask_classes=np.unique(new_mask)
    print(f"mask id include :{mask_classes}")

    # refresh the data of roi_layer and mask_layer
    _on_refresh_roi2(new_mask,new_roi)




 

@sub_viewer.mouse_double_click_callbacks.append
def on_double_click_at_sub_viewer(_module,event):
    print(f'double click at sub_viewer: type of _moudule')

    #cursor coords(3d) -> data coords(2d) + slice_idx -->roi_center coords -->roi_offset+roi_size--->roi
    #rescure the old roi hint in aux_slice,
    #draw new roi hint in aux_slice
    mouse_pos=sub_viewer.cursor.position
    global apply_roi_mip 
    global two_dim_downsample_level
    
    if any(key.name.startswith('aux') for key in sub_viewer.layers): 
        
        for key in sub_viewer.layers:
            if key.name.startswith('aux_slice_mask'):
                slice_idx = int(key.name.split('aux_slice_mask')[-1])

        data_coords=sub_viewer.layers[f'aux_slice{slice_idx}'].world_to_data(mouse_pos)
        data_coords=np.round(data_coords).astype(int)
        data_coords = data_coords * (2**two_dim_downsample_level)

        roi_center_idx=np.insert(data_coords,0,slice_idx)
        roi_offset=roi_center_idx - roi_size//2
        roi_offset = np.maximum(roi_offset, 0)
        print(f"roi_offset:{roi_offset},roi_size:{roi_size}")

        # roi=ims_vol.from_roi(coords=[*roi_offset, *roi_size],level=level)
        # print(f"roi start at {roi_offset} with shape{roi_size}")
        # roi=roi.reshape(*np.squeeze(roi_size))
        # # mask=get_mask_from_different_scale(lr_mask,roi_offset,roi_size,zoom_factor)
        # roi = mip(roi,apply_roi_mip)
        # print(f"apply mip is {apply_roi_mip}, after mip, roi_size is {roi.shape}")

        # target_roi_size = roi.shape
        # if len(roi.shape) == 2:
        #     target_roi_size = np.insert(target_roi_size,0,1)

        # mask=get_mask_from_different_scale(lr_mask,roi_offset,target_roi_size,zoom_factor)
        roi, mask = get_roi_and_mask_mip_defined_indexes(roi_size,roi_offset,level,zoom_factor,apply_roi_mip)
        

        #refresh polygen that indicate roi
        idx_ = [ int( idx //(2**two_dim_downsample_level)) for idx in roi_offset]
        roi_ =  [max(1,int(edge//(2**two_dim_downsample_level))) for edge in roi_size]
        roi_polygon = np.array( 
        [
            [idx_[1]           , idx_[2]],
            [idx_[1] + roi_[1] , idx_[2]],
            [idx_[1] + roi_[1] , idx_[2] + roi_[2]],
            [idx_[1]           , idx_[2] + roi_[2]],
        ])
        sub_viewer.layers['polygon'].data = roi_polygon
        
        _on_refresh_roi2(mask,roi)
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


if __name__ == '__main__':
    napari.run()