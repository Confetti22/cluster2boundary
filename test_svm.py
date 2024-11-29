
import os
import sys
sys.path.insert(0,f'{os.getcwd()}/napari_helper')

from napari_helper._svc_widget import SvcWidget
from napari_helper.napari_view_utilis import link_pos_of_plane_layers,remove_layers_with_patterns
from config.constants import config
import napari
import tifffile as tif
from napari_threedee.annotators import PointAnnotator

viewer=napari.Viewer(ndisplay=3)
roi=tif.imread('/home/confetti/workspace/cluster2boudnary/new_t11/test_data/roi1/roi.tif')
mask=tif.imread('/home/confetti/workspace/cluster2boudnary/new_t11/test_data/roi1/mask.tif')

roi_plane_parameters = config["roi_plane_parameters"]
mask_plane_parameters = config["mask_plane_parameters"]


roi_layer=viewer.add_image(roi,name='roi')
mask_layer=viewer.add_labels(mask,name='mask')
points_layer=viewer.add_points(data=[],name='points',size=2,face_color='white',ndim=3,blending='additive')




roi_plane_layer=viewer.add_image(roi, name='roi_plane', depiction='plane',rendering='mip',  blending='additive', opacity=0.6, plane=roi_plane_parameters)
mask_plane_layer=viewer.add_labels(mask, name='mask_plane', depiction='plane',blending='additive', opacity=0.6, plane=mask_plane_parameters,)


link_pos_of_plane_layers([roi_plane_layer,mask_plane_layer])
viewer.layers.selection=[roi_plane_layer]

annotator = PointAnnotator(
        viewer=viewer,
        image_layer=roi_plane_layer,
        mask_layer=mask_plane_layer,
        points_layer=points_layer,
        enabled=True,
        config=config
    )


svc_predictor=SvcWidget(viewer,points_layer,mask_layer,config=config)


if __name__ == '__main__':
    napari.run()