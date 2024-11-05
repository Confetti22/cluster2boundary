import numpy as np
from scipy import ndimage as ndi
from skimage import data
import os
import napari
import tifffile as tif
from pathlib import Path
from read_ims import get_random_roi
from skimage.measure import shannon_entropy
from read_ims import get_2d_slices

def get_roi(data, cords, radius):
    """
    Extracts a region of interest (ROI) from the data.

    Parameters:
    - data: np.ndarray, 2D or 3D array from which to extract the ROI.
    - cords: tuple, coordinates (x, y [,z]) for the center of the ROI.
    - radius: int, radius of the ROI to extract.

    Returns:
    - np.ndarray, the extracted region of interest.
    """
    # Ensure the cords are within bounds
    cords = np.array(cords)
    radius = np.array([radius]*len(cords))

    # Define the slicing boundaries
    min_cords = np.maximum(cords - radius, 0)
    max_cords = np.minimum(cords + radius + 1, np.array(data.shape))

    # Create the slicing tuple
    slices = tuple(slice(min_cords[i], max_cords[i]) for i in range(len(cords)))

    return data[slices]

os.environ['DISPLAY'] = ':1'
viewer = napari.Viewer(ndisplay=3)

def entropy_filter(thres=1.2):
    def _filter(img):
        entropy=shannon_entropy(img)
        return entropy>thres
    return _filter

sigma=3
radius=4
entropy_thres=1.8 #for entrop_map thres-holding

ims_pth="/home/confetti/data/visor/T11.ims.h5"

viewer = napari.Viewer(ndisplay=3)


roi=get_random_roi(ims_pth,2,entropy_filter())


images_layer=viewer.add_image(roi,contrast_limits=(0,np.percentile(roi,99)),colormap='gray',name='roi')





@viewer.mouse_drag_callbacks.append
def get_event(viewer, event):
    print(f"function1:{event}")

  
      

# Handle click or drag events separately
@images_layer.mouse_drag_callbacks.append
def click_drag(layer, event):
    print('function3 mouse down')
    dragged = False
    yield
    # on move
    while event.type == 'mouse_move':
        print(event.position)
        dragged = True
        yield
    # on release
    if dragged:
        print('function3 drag end')
    else:
        print('function3 clicked!')


@viewer.mouse_drag_callbacks.append
def get_local_entropy(layer, event,radius=radius):
    # layer=layer.layers[0]
    layer=viewer.layers.selection.active
    data_coordinates = layer.world_to_data(event.position)
    cords = np.round(data_coordinates).astype(int)
    val = layer.get_value(data_coordinates)
    if val is None:
        return
    else:
        data = layer.data
        roi=get_roi(data, cords, radius)
        entropy=shannon_entropy(roi)
        print(f'entropy at {cords} of win_size{radius*2+1}is {entropy}')

# Handle click or drag events separately
@viewer.mouse_double_click_callbacks.append
def on_second_click_of_double_click(layer, event):
    print('Second click of double_click fectch new roi')
    new_roi=get_random_roi(ims_pth,2,entropy_filter())
    if "roi" in viewer.layers:
        viewer.layers.remove("roi")
    viewer.add_image(new_roi,contrast_limits=(0,np.percentile(new_roi,100)),name="roi")


if __name__ == '__main__':
    napari.run()