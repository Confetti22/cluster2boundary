import tifffile as tif
import napari
import numpy as np

from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

def get_binary_mask(parent_acronym,mask,bg_atlas):
    #change all the descendent id belonogs to target_id to target_id
    #change into binary mask
    """
    Convert all descendants of a given structure in the mask volume to the ID of the parent structure.

    Parameters:
        mask (numpy.ndarray): The mask volume to modify.
        bg_atlas: The BrainGlobe Atlas instance.
        parent_acronym (str): The acronym of the parent structure (e.g., "fiber tracts").
    """
    
    
    descendant_acronyms = bg_atlas.get_structure_descendants(parent_acronym)
    parent_id = bg_atlas.structures[parent_acronym]['id']
    
    # Get IDs for all descendants
    descendant_ids = [bg_atlas.structures[desc]["id"] for desc in descendant_acronyms]
    
    # Create a copy of the mask to modify
    modified_mask = np.zeros_like(mask)
    
    # Replace all descendant IDs in the mask with the parent ID
    for desc_id in descendant_ids:
        modified_mask[mask == desc_id] = 1 

    #for parent_acronym
    modified_mask[mask == parent_id] = 1
    
    
    return modified_mask

def get_modified_mask(acronym_lst,mask,half=False):
    if half:
        half_x = int((mask.shape[-1])//2)
        mask[:,:,half_x:] = 0
    if acronym_lst == None:
        return mask
    atlas = BrainGlobeAtlas("allen_mouse_25um")
    modified_mask = np.zeros_like(mask)
    for idx, acronym in enumerate(acronym_lst):
        binary_mask = get_binary_mask(acronym,mask,atlas)
        modified_mask[binary_mask == 1] = idx+1
    return modified_mask

def compute_cuboid_roi(mask, margin_ratio=0.2):
    """
    Compute the offset and ROI size of a cuboid that wraps a 3D binary mask with a margin.
    
    Parameters:
        mask (numpy.ndarray): A binary 3D mask of shape (Z, Y, X).
        margin_ratio (float): The margin ratio relative to the ROI size in each dimension.
        
    Returns:
        offset (tuple): The starting coordinate (z_offset, y_offset, x_offset) of the cuboid.
        roi_size (tuple): The size (z_size, y_size, x_size) of the cuboid.
    Example:
        >>> import numpy as np
        >>> mask = np.zeros((100, 100, 100), dtype=np.uint8)
        >>> mask[30:70, 40:80, 50:90] = 1  # Example cuboid mask
        >>> offset, roi_size = compute_cuboid_roi(mask)
        >>> offset
        (24, 34, 44)
        >>> roi_size
        (52, 52, 52)
    """
    # Get the coordinates of non-zero elements in the mask
    coords = np.argwhere(mask > 0)
    
    # Calculate the bounding box of the mask
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1  # Add 1 to include the boundary
    
    # Compute the size of the bounding box
    bbox_size = max_coords - min_coords
    
    # Calculate the margin in each direction
    margin = np.ceil(margin_ratio * bbox_size).astype(int)
    
    # Compute the offset and size of the ROI with margin
    offset = np.maximum(min_coords - margin, 0)  # Ensure non-negative offset
    roi_size = bbox_size + 2 * margin  # Add margin on both sides
    
    return tuple(offset), tuple(roi_size)



if __name__ == '__main__':
    atlas = BrainGlobeAtlas("allen_mouse_25um")
    # print(atlas.structures[603])
    # print(atlas.get_structure_descendants("fiber tracts"))


    mask = tif.imread("/home/confetti/mnt/data/processed/t1779/r32_ims_downsample_561_register/registered_atlas.tiff")
    raw = tif.imread('/home/confetti/mnt/data/processed/t1779/r32_ims_downsample_561_register/downsampled.tiff')
    ft_mask = get_modified_mask(['fi'],mask)

    #TODO check the radius of most fiber_tract region
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_labels(ft_mask,opacity=0.27)
    viewer.add_image(raw)
    napari.run()

