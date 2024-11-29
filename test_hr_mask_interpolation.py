#%%
from scipy.ndimage import zoom
from config.constants import config
from napari_helper.read_ims import Ims_Image
import tifffile as tif
import numpy as np
from skimage.measure import shannon_entropy
import matplotlib.pyplot as plt
def get_hr_mask(lr_mask,indexs,roi_size,scale,zoom_order=0):
    """
    roi is defined via indexs +roi_size
    """
    lr_indexs=[int(idx/scale)for idx in indexs]
    # lr_roi_size=[int(roi/scale)for roi in roi_size]
    lr_roi_size = [max(int(roi / scale), 1) for roi in roi_size]
    z,y,x=lr_indexs
    z_size,y_size,x_size=lr_roi_size

    print(f"hr_index:{indexs}, lr_index:{lr_indexs}")
    print(f"lr_roi_size:{lr_roi_size}")
    print(f"target_roi_size:{roi_size}")

    #mask in order (z,y,x)
    lr_mask_roi=lr_mask[z:z+z_size,y:y+y_size,x:x+x_size]
    zoom_factors=[t/s for t,s in zip(roi_size,lr_roi_size)]
    zoomed_mask_roi=zoom(lr_mask_roi,zoom=zoom_factors,order=zoom_order)
    zoomed_mask_roi=np.squeeze(zoomed_mask_roi)


    return lr_mask_roi,zoomed_mask_roi

def upsample_mask(mask1, scale_factor):
    # Get unique labels
    unique_labels = np.unique(mask1)
    
    # Prepare the high-resolution mask
    high_res_shape = (np.array(mask1.shape) * np.array(scale_factor)).astype(int)
    high_res_mask = np.zeros(high_res_shape, dtype=np.int32)
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        # Create binary mask for the current label
        binary_mask = (mask1 == label).astype(np.float32)
        
        # Upsample the binary mask using zoom
        upsampled_binary = zoom(binary_mask, scale_factor, order=1)

 
        # Use watershed segmentation to refine boundaries
        markers = (upsampled_binary > 0.5).astype(np.int32)
        # distance = distance_transform_edt(upsampled_binary > 0.5)
        # refined_region = watershed(-distance, markers, mask=(upsampled_binary > 0))
        # high_res_mask[refined_region > 0] = label
        high_res_mask[markers > 0] = label
    
    return high_res_mask

def refined_get_hr_mask(lr_mask,indexs,roi_size,scale):
    lr_indexs=[int(idx/scale)for idx in indexs]
    # lr_roi_size=[int(roi/scale)for roi in roi_size]
    lr_roi_size = [max(int(roi / scale), 1) for roi in roi_size]
    z,y,x=lr_indexs
    z_size,y_size,x_size=lr_roi_size
    lr_mask_roi=lr_mask[z:z+z_size,y:y+y_size,x:x+x_size]
    zoom_factors=[t/s for t,s in zip(roi_size,lr_roi_size)]

    zoomed_msk=upsample_mask(lr_mask_roi,zoom_factors)
    return zoomed_msk




#%%

img_pth = config["image_paths"]["img_pth"]
mask_pth = config["image_paths"]["mask_pth"]
level = config["level"]
roi_size = (1,56,56) 
zoom_factor = config["zoom_factor"]
roi_plane_parameters = config["roi_plane_parameters"]
mask_plane_parameters = config["mask_plane_parameters"]

def entropy_filter(thres=1.8):
    def _filter(img):
        entropy=shannon_entropy(img)
        return entropy>thres
    return _filter

# Read IMS image and TIFF mask
ims_vol = Ims_Image(img_pth, channel=3)
lr_mask = tif.imread(mask_pth)
hr_vol_shape = ims_vol.info[level]['data_shape']
zoom_order=0
print(f"zoom_order{zoom_order}")
#%%
#visualize the lr_mask and zoomed_mask
# in lr_mask(original resolution), zigzag boundary will often take up 1 pixel
# but in zoomed_mask, these zigzag will also be zoomed in to take up more pixels


 
# nrow=8
# ncols=3
# plt.figure(figsize=(12,64))
# for row in range(nrow):
#     new_roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(thres=1.6),roi_size=roi_size,level=0)
#     lr_mask_roi,zoomed_mask_roi=get_hr_mask(lr_mask,indexs,roi_size=roi_size,scale=zoom_factor,zoom_order=zoom_order  )
#     refined_zoomed=refined_get_hr_mask(lr_mask,indexs,roi_size=roi_size,scale=zoom_factor)
#     lr_mask_roi=lr_mask_roi.squeeze()
#     zoomed_mask_roi=zoomed_mask_roi.squeeze()
#     refined_zoomed=refined_zoomed.squeeze()
#     plt.subplot(nrow,ncols,ncols*row+1)
#     plt.title('lr_mask_roi')
#     plt.imshow(lr_mask_roi,cmap='jet',interpolation=None)

#     plt.subplot(nrow,ncols,ncols*row+2)
#     plt.title('zoomed_mask_roi')
#     plt.imshow(zoomed_mask_roi,cmap='jet',interpolation=None)

#     plt.subplot(nrow,ncols,ncols*row+3)
#     plt.title('refined_zoomed')
#     plt.imshow(refined_zoomed,cmap='jet',interpolation=None)
    
# plt.show()

import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass

def annotate_labeled_regions(image, labels, ax=None, text_color='red', fontsize=12):
    """
    Annotates the centers of labeled regions on a given image.
    """
    if ax is None:
        ax = plt.gca()  # Get the current axis if none is provided
    
    for label in labels:
        if label == 0:  # Skip the background if labeled as 0
            continue
        center = center_of_mass(image == label)
        ax.text(center[1], center[0], str(label), color=text_color, fontsize=fontsize, 
                ha='center', va='center', fontweight='bold')

nrow, ncols = 8, 3
plt.figure(figsize=(12, 64))

cnt=0
while cnt != nrow:
    new_roi, indexs = ims_vol.get_random_roi(filter=entropy_filter(thres=1.6), roi_size=roi_size, level=0)
    lr_mask_roi, zoomed_mask_roi = get_hr_mask(lr_mask, indexs, roi_size=roi_size, scale=zoom_factor, zoom_order=zoom_order)

    if len(np.unique(lr_mask_roi)) >= 2:
        cnt = cnt + 1
        row = cnt - 1
        refined_zoomed = refined_get_hr_mask(lr_mask, indexs, roi_size=roi_size, scale=zoom_factor)
        
        lr_mask_roi = lr_mask_roi.squeeze()
        zoomed_mask_roi = zoomed_mask_roi.squeeze()
        refined_zoomed = refined_zoomed.squeeze()
        
        # Plot low-resolution mask ROI
        plt.subplot(nrow, ncols, ncols * row + 1)
        plt.title('lr_mask_roi')
        plt.imshow(lr_mask_roi, cmap='jet', interpolation=None)
        annotate_labeled_regions(lr_mask_roi, labels=np.unique(lr_mask_roi))  # Annotate labels
        
        # Plot zoomed mask ROI
        plt.subplot(nrow, ncols, ncols * row + 2)
        plt.title('zoomed_mask_roi')
        plt.imshow(zoomed_mask_roi, cmap='jet', interpolation=None)
        annotate_labeled_regions(zoomed_mask_roi, labels=np.unique(zoomed_mask_roi))  # Annotate labels
        
        # Plot refined zoomed ROI
        plt.subplot(nrow, ncols, ncols * row + 3)
        plt.title('refined_zoomed')
        plt.imshow(refined_zoomed, cmap='jet', interpolation=None)
        annotate_labeled_regions(refined_zoomed, labels=np.unique(refined_zoomed))  # Annotate labels


# %%
nrow=8
plt.figure(figsize=(4,32))
shape=lr_mask.shape
roi_size=(1,32,32)

mask_list=[]
mask_indexs=[]
cnt=0
while cnt != nrow:
    z_idx=np.random.randint(0,shape[0]-roi_size[0]) 
    y_idx=np.random.randint(0,shape[1]-roi_size[1]) 
    x_idx=np.random.randint(0,shape[2]-roi_size[2]) 
    mask_roi=lr_mask[z_idx:z_idx+roi_size[0],y_idx:y_idx+roi_size[1],x_idx:x_idx+roi_size[2]]
    mask_roi=mask_roi.squeeze()
    if len(np.unique(mask_roi))>=2:
        counts = np.bincount(mask_roi.flatten())
        total = np.sum(counts)
        label_ratio=counts/total
        cnt+=1
        mask_list.append(mask_roi)
        mask_indexs.append((z_idx,y_idx,x_idx))
        plt.subplot(nrow,1,cnt)
        plt.title(f"index{cnt-1}")
        plt.imshow(mask_roi,cmap='jet',interpolation=None)
plt.show()

idx_list=[4]


# %%
idx_list=[6]
dir_2="/home/confetti/workspace/cluster2boudnary/new_t11/test_data/test_mask_rois_2classes"
dir_m="/home/confetti/workspace/cluster2boudnary/new_t11/test_data/test_mask_rois_multiclasses"
dir=dir_m
for idx in idx_list:
    mask=mask_list[idx]
    indexs=mask_indexs[idx]
    tif.imwrite(f"{dir}/{indexs[0]}_{indexs[1]}_{indexs[2]}.tif",mask)
# %%
