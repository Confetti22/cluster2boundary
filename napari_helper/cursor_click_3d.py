
import numpy as np

import napari
from read_ims import Ims_Image
from skimage.measure import shannon_entropy

n_points = 16


points = np.zeros(shape=(n_points, 3))

# point sizes
point_sizes = np.linspace(7, 4, n_points, endpoint=True)

# point colors
green = [0, 1, 0, 1]
magenta = [1, 0, 1, 1]
point_colors = np.linspace(green, magenta, n_points, endpoint=True)

# create viewer and add layers for each piece of data
viewer = napari.Viewer(ndisplay=3)

ray_layer = viewer.add_points(
    points, face_color=point_colors, size=point_sizes, name='cursor ray'
)
def entropy_filter(thres=1.2):
    def _filter(img):
        entropy=shannon_entropy(img)
        return entropy>thres
    return _filter
roi_size=[64,64,64]

img_pth="/home/confetti/mnt/data/processed/new_t11/T11.ims.h5"
ims_vol=Ims_Image(img_pth,channel=3)
# roi,_=ims_vol.get_random_roi(filter=entropy_filter(thres=1.6),roi_size=roi_size,level=0)
slice=ims_vol.from_slice(300,0,0)
slice=slice.reshape(1,slice.shape[0],slice.shape[1])
cell_layer=viewer.add_image(slice,name='slice')


# callback function, called on mouse click when volume layer is active
@cell_layer.mouse_drag_callbacks.append
def on_click(layer, event):
    print(f"click at {event.position}")
    print(f"on slice the pos is {cell_layer.world_to_data(event.position)}")
    near_point, far_point = layer.get_ray_intersections(
        event.position,
        event.view_direction,
        event.dims_displayed
    )
    print(f"near_p:{near_point} .    far_p:{far_point}")
    if (near_point is not None) and (far_point is not None):
        ray_points = np.linspace(near_point, far_point, n_points, endpoint=True)
        if ray_points.shape[1] != 0:
            ray_layer.data = ray_points







if __name__ == '__main__':
    napari.run()