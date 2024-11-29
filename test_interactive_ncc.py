import numpy as np
from confettii.ncc_helper import get_ncc_point_to_whole,crop_2d_image
from confettii.entropy_helper import entropy_filter
from feature_extract.feature_extractor import get_feature_list, get_encoder
from PIL import Image
import pickle
from napari_helper.napari_view_utilis import toggle_layer_visibility,remove_layers_with_patterns
from scipy.ndimage import zoom

import tifffile as tif
import torch

import napari



img_pth="/home/confetti/data/mousebrainatlas/histology/102117913_d0.png"
img=Image.open(img_pth)
img=np.array(img)

roi_size=600
roi=crop_2d_image(img,ori=(1660,2880),range=(roi_size,roi_size))
#cannot direct downsample to 1/4, will lose mophlogy info

viewer=napari.Viewer(ndisplay=2)
viewer.add_image(roi)

feats_pth="/home/confetti/e5_workspace/cluster2boudnary/feats_roi100_resnet18.pkl"
with open(feats_pth,'rb') as file:
    feats_lst=pickle.load(file)



feats_map=feats_lst.reshape(roi_size,roi_size,-1)
ncc_map=get_ncc_point_to_whole((408,425),feats_map)
viewer.add_image(ncc_map,colormap='viridis',name="ncc_map")


@viewer.mouse_double_click_callbacks.append
def compute_ncc_on_click(viewer,event):

    print(f"double_click")
    mouse_pos=viewer.cursor.position
    data_coords=viewer.layers['roi'].world_to_data(mouse_pos)
    data_coords=np.round(data_coords).astype(int)
    print(f"data_coords{data_coords}")
    ncc_map=get_ncc_point_to_whole(data_coords,feats_map)

    remove_layers_with_patterns(viewer.layers,['ncc_map'])

    viewer.add_image(ncc_map,colormap='viridis',name='ncc_map_viridis')

@viewer.bind_key('v')
def toggle_ncc_map(_module):
    print(f"press v at viewer")
    toggle_layer_visibility(layers=viewer.layers,name_patterns=['ncc_map'])


if __name__ == '__main__':
    napari.run()