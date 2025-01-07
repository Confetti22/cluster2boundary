from scipy.ndimage import center_of_mass,zoom
from skimage.morphology import convex_hull_image
import napari
import numpy as np
import tifffile as tif
import time

from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
atlas = BrainGlobeAtlas("allen_mouse_25um")



def find_inside_centroid(mask, label_value):
    # Extract the region for the specific label
    region = (mask == label_value)
    
    # Compute the initial center of mass
    candidate_center = center_of_mass(region)
    
    # Find all coordinates within the region
    region_coords = np.argwhere(region)
    
    # Compute the Euclidean distance from the candidate to all region points
    distances = np.linalg.norm(region_coords - candidate_center, axis=1)
    
    # Find the region coordinate closest to the candidate center
    closest_idx = np.argmin(distances)
    return region_coords[closest_idx]

def _compute_annotation(mask,acronym = False,verbose = True):


    current = time.time()
    label_indices = np.unique(mask)[1:] #skip background(label 0)
    centers = np.array([find_inside_centroid(mask, label) for label in label_indices])

    if acronym:
        text_annotations_lst = [f"{atlas.structures[label]['acronym']}"for label in label_indices]
    else:
        text_annotations_lst = [f"{int(label)}"for label in label_indices]

    if verbose:
        print(f"compute annotation : {time.time() - current}")


    return centers,text_annotations_lst

def compute_annotation(mask,acronym=False,verbose = False):
    #make sure shape of mask is two dim
    middle_width = int(mask.shape[-1])//2
    l_centers, l_annotations = _compute_annotation(mask[:,:middle_width],acronym,verbose)
    r_centers, r_annotations = _compute_annotation(mask[:,middle_width:],acronym,verbose)
    r_centers[:,1] += middle_width #r_centers need a horizontal shift
    
    centers = np.concatenate((l_centers,r_centers),axis=0 )
    annotations = l_annotations + r_annotations
    return centers, annotations

if __name__ == "__main__":
    current = time.time()
    mask = tif.imread("/home/confetti/mnt/data/processed/t1779/r32_ims_downsample_561_register/registered_atlas.tiff")
    mask = mask[int(mask.shape[0]//2),:,:]

    print(f"load mask : {time.time() -current}, mask shape {mask.shape}")
    current = time.time()

        
    centers,text_annotations = compute_annotation(mask,acronym=True)

    viewer = napari.Viewer(ndisplay=2)
    viewer.add_labels(mask,name = 'mask')
    viewer.add_points(
        centers,
        properties={"label": text_annotations},
        text={
            "string": "{label}",
            "anchor": "center",
            "color": "white",
            "size": 6,
        },
        size=4,
        face_color="transparent",
        edge_color= 'transparent',
        name="region_annotation_id",
    )


    napari.run()