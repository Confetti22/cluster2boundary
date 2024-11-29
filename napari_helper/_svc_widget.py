from typing import Any, Generator, Optional
from napari.utils.events import Event
import napari
import napari.layers
import numpy as np
import torch
from magicgui.widgets import ComboBox, Container, PushButton, create_widget
from napari.layers import Image, Points, Shapes
from napari.layers.shapes._shapes_constants import Mode

from qtpy.QtCore import Qt
from _build_svc import svc_model_registry , SvcPredictor
from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
from segment_anything.modeling import Sam
from skimage import color, util
import pprint
from napari_segment_anything.utils import get_weights_path
from napari_view_utilis import _filter_layer_name_with_pattern, remove_layers_with_patterns,toggle_layer_visibility

from typing import List
import sys
import pprint
            

def point_color_in_mask(color,mask_layer:napari.layers.Labels):
    #determine the rgb part in a rgba is in a rgba list
    rgba_lst=[mask_layer.get_color(id) for id in np.unique(mask_layer.data)]
    # Extract only the RGB components from each RGBA value in the list
    rgb_lst = [rgba[:3] for rgba in rgba_lst]
    
    # Convert the given color to a numpy array for easy comparison
    color = np.array(color)
    color=color[:3]
    # Check if the given RGB color is in the RGB list
    return any(np.allclose(color, rgb) for rgb in rgb_lst)

def get_dims_displayed(layer):
    # layer._dims_displayed was removed in
    # https://github.com/napari/napari/pull/5003
    if hasattr(layer, "_slice_input"):
        return layer._slice_input.displayed
    return layer._dims_displayed


def point_in_layer_bounding_box(point, layer):
    """Determine whether an nD point is inside a layers nD bounding box.

    Parameters
    ----------
    point : np.ndarray
        (n,) array containing nD point coordinates to check.
    layer : napari.layers.Layer
        napari layer to get the bounding box from
    
    Returns
    -------
    bool
        True if the point is in the bounding box of the layer,
        otherwise False
    
    Notes
    -----
    For a more general point-in-bbox function, see:
        `napari_threedee.utils.geometry.point_in_bounding_box`
    """
    dims_displayed = get_dims_displayed(layer)
    bbox = layer._display_bounding_box(dims_displayed).T
    if np.any(point < bbox[0]) or np.any(point > bbox[1]):
        print(f"pt{point} without bbox{bbox[0]} -- {bbox[1]}")
        return False
    else:
        print(f"pt{point} is in bbox{bbox[0]} -- {bbox[1]}")
        return True



class SvcWidget(Container):
    """
    pts_layer: user points prompt input
    """

    def __init__(
            self, 
            viewer: napari.Viewer,
            sub_viewer: napari.Viewer,
            pts_layer:napari.layers.Points, 
            mask_layer:napari.layers.Labels,
            model_type:str = "default",
            config:dict = None
            ):
        super().__init__()

        self._viewer = viewer
        self.sub_viewer=sub_viewer

        self._pts_layer=pts_layer #for getting user annotation points
        self._pts_layer.events.data.connect(self._on_pts_change_run)
        # self._pts_layer.events.face_color.connect(self._on_pts_change_run)

        self._mask_layer=mask_layer #for getting pesudo label

        self._temp_mask=self._mask_layer.data.copy()

        self._config=config

        #put this _cdic init in the preprocess method
        # _cdic: a colormap{ rgb : label_id }
        # self._cdic = {
        #     tuple(self._mask_layer.get_color(id)[:3] if self._mask_layer.get_color(id) is not None else [1, 1, 1]): id 
        #     for id in np.unique(self._mask_layer.data)
        # }

        self._load_model()
        # self._model_type_widget = ComboBox(
        #     value=model_type,
        #     choices=list(svc_model_registry.keys()),
        #     label="Model:",
        # )

  
        # self._confirm_mask_btn = PushButton(
        #     text="Confirm Annot.",
        #     enabled=False,
        #     tooltip="Press C to confirm annotation.",
        # )

        # self._confirm_mask_btn.changed.connect(self._on_confirm_mask)

        # self.append(self._confirm_mask_btn)
       
        # self._cancel_annot_btn = PushButton(
        #     text="Cancel Annot.",
        #     enabled=False,
        #     tooltip="Press X to cancel annotation.",
        # )
        # self._cancel_annot_btn.changed.connect(self._cancel_annot)
        # self.append(self._cancel_annot_btn)
        
        # self._model_type_widget.changed.emit(model_type)



    def _get_indexs_from_roi_size(self):
        z_indices, y_indices, x_indices = np.indices(self._config['roi_size'])
        point_indexs = np.stack((z_indices.ravel(), y_indices.ravel(), x_indices.ravel()), axis=-1)
        return point_indexs

    def _preprocess_anno_points(self)->tuple[List,List[int]]:
        """
        convert color of points to label 
        filter out the points lies outside the bbox of roi  or  the color is not within the color range of mask
        return :
        points: a list of 3d points
        anno_labels: a list of label
        """

        # _cdic: a colormap{ rgb : label_id }
        self._cdic = {
            tuple(self._mask_layer.get_color(id)[:3] if self._mask_layer.get_color(id) is not None else [1, 1, 1]): id 
            for id in np.unique(self._mask_layer.data)
        }

        points=[]
        anno_labels=[]
        for point,color in zip(self._pts_layer.data,self._pts_layer.face_color):
            if not point_in_layer_bounding_box(point,self._mask_layer):
                continue
            if not point_color_in_mask(color,self._mask_layer):
                continue 
            points.append(point)
            anno_labels.append(self._cdic[tuple(color[0:3])])
        point=np.array(point)
        anno_labels=np.array(anno_labels)
        return points,anno_labels

    def _annotate_volume_within_spheres_around_pts( self,annotation_points, labels, radius)->np.ndarray:
        """
        it's function has been validated!
        Annotates a 3D volume with labels at specified points, applying labels within a sphere around each point.

        Parameters:
        - volume: 3D numpy array of shape (Z, Y, X) representing the origin label volume.
        - annotation_points: List of annotation points where each point is a tuple or list (z, y, x).
        - labels: List of labels corresponding to each annotation point, same length as annotation_points.
        - radius: Radius (in pixels) for annotating vicinity points.

        Returns:
        - Annotated 3D volume with updated labels.

        """

        volume=self._mask_layer.data.copy()
        # Ensure radius is a float for calculations
        radius = float(radius)
        
        # Get shape of the original volume
        Z, Y, X = volume.shape

        # Create a meshgrid to compute the distance to the annotation points
        zz, yy, xx = np.meshgrid(
            np.arange(Z), np.arange(Y), np.arange(X), indexing='ij'
        )

        
        for (z, y, x), label in zip(annotation_points, labels):
            # Calculate the distance of each voxel from the current annotation point
            distance = np.sqrt((zz - z)**2 + (yy - y)**2 + (xx - x)**2)
            
            # Create a mask for all points within the sphere radius
            sphere_mask = distance <= radius
            
            # Apply the label to all points within the sphere
            volume[sphere_mask] = label

        return volume


    def _load_model(self) -> None:
        paras=self._config["svc_parameters"]

        gamma=paras["gamma"]
        C=paras["C"]
        kernel=paras["kernel"]
        classweight=paras["classweight"]
        norm=paras["norm"]

        self._predictor = SvcPredictor(gamma=gamma,\
                                       C=C,\
                                       kernel=kernel,\
                                       classweight=classweight,\
                                       norm=norm,\
                                       config=self._config)

    
 
    #event pts_layer is change
    # for each label_point in pts_layer:
    # filter out the points out of the range of roi and rgba label 
    # mask the adjacent points in mask_roi to the same label of label_point
    # train_svm

    
    def _on_pts_change_run(self, event:Event,trigger_points_num=5) -> None:
        print(f"\n \n _on_pts_change: event is {event.source}")
        points = self._pts_layer.data
        print(f"condition:{len(points) % trigger_points_num != 0}")

        if  len(points) == 0 or  (len(points) % trigger_points_num != 0):
            return
 


        remove_layers_with_patterns(self._viewer.layers,['svc','train_data'])
        #for points layer, coords in data and world are the same
        points,anno_labels=self._preprocess_anno_points()

        new_mask=self._annotate_volume_within_spheres_around_pts( points, anno_labels, radius=4)
        


        # print(f'type of classes before anno are{np.unique(self._mask_layer.data)}')

        svc_pred_mask= self._predictor.findboundary(
            X=self._get_indexs_from_roi_size(),
            Y=new_mask.ravel()
        )

        #TODO :update svc_pred_mask in the sub_viewer:
        # z_range=svc_pred_mask.shape[0]
        # svc_pred_mid_plane=svc_pred_mask[int(z_range//2),:,:]
        # sub_anno_layer_name=_filter_layer_name_with_pattern(self.sub_viewer.layers,name_patterns=['aux_slice_mask'])
        # ori_sub_anno=self.sub_viewer.layers[sub_anno_layer_name].data
        # updated_sub_anno = ori_sub_anno.copy()
        # updated_sub_anno 



        # print(f"svc_pred_mask={svc_pred_mask}")

        self._viewer.add_labels(svc_pred_mask,name='svc_mask_plane',depiction='plane',blending='additive',opacity=0.7,plane=self._config['mask_plane_parameters'])
        self._viewer.add_labels(svc_pred_mask,name='svc_mask',visible=False)
        self._viewer.layers.selection=[self._viewer.layers['roi_plane']]




  

    def _on_confirm_mask(self, _: Optional[Any] = None) -> None:
        # if self._image is None:
        #     return

        # labels = self._svc_labels_layer.data
        # mask = self._mask_layer.data
        # labels[np.nonzero(mask)] = labels.max() + 1
        # self._svc_labels_layer.data = labels
        # self._cancel_annot()
        print(f"pressed confirm_mask")

    def _cancel_annot(self, _: Optional[Any] = None) -> None:
        # boxes must be reset first because of how of points data update signal
        # self._pts_layer.data = []
        # self._mask_layer.data = np.zeros_like(self._mask_layer.data)

        # self._confirm_mask_btn.enabled = False
        # self._cancel_annot_btn.enabled = False
        print(f"press cancel annotation")