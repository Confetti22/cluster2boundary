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
num=8
plt.figure(figsize=(8,64))
for row in range(num):
    new_roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(thres=1.6),roi_size=roi_size,level=0)
    lr_mask_roi,zoomed_mask_roi=get_hr_mask(lr_mask,indexs,roi_size=roi_size,scale=zoom_factor,zoom_order=zoom_order  )
    lr_mask_roi=lr_mask_roi.squeeze()
    zoomed_mask_roi=zoomed_mask_roi.squeeze()
    plt.subplot(num,2,2*row+1)
    plt.title('lr_mask_roi')
    plt.imshow(lr_mask_roi,cmap='jet',interpolation=None)
    plt.subplot(num,2,2*row+2)
    plt.title('zoomed_mask_roi')
    plt.imshow(zoomed_mask_roi,cmap='jet',interpolation=None)
    
plt.show()
# %%
num=8
plt.figure(figsize=(4,32))
shape=lr_mask.shape
roi_size=(1,32,32)

mask_list=[]
mask_indexs=[]
cnt=0
while cnt != num:
    z_idx=np.random.randint(0,shape[0]-roi_size[0]) 
    y_idx=np.random.randint(0,shape[1]-roi_size[1]) 
    x_idx=np.random.randint(0,shape[2]-roi_size[2]) 
    mask_roi=lr_mask[z_idx:z_idx+roi_size[0],y_idx:y_idx+roi_size[1],x_idx:x_idx+roi_size[2]]
    mask_roi=mask_roi.squeeze()
    if len(np.unique(mask_roi))==2:
        counts = np.bincount(mask_roi.flatten())
        total = np.sum(counts)
        label_0_ratio = counts[0] / total
        label_1_ratio = counts[1] / total
        rr=[0,1]
        if rr[0] <= label_0_ratio <= rr[1] and rr[0]<= label_1_ratio <= rr[1]:
            cnt+=1
            mask_list.append(mask_roi)
            mask_indexs.append((z_idx,y_idx,x_idx))
            plt.subplot(num,1,cnt)
            plt.title(f"index{cnt-1}")
            plt.imshow(mask_roi,cmap='jet',interpolation=None)
plt.show()

idx_list=[4]


# %%
idx_list=[6]
dir_2="/home/confetti/workspace/cluster2boudnary/new_t11/test_data/test_mask_rois_2classes"
dir_m="/home/confetti/workspace/cluster2boudnary/new_t11/test_data/test_mask_rois_multiclasses"
dir=dir_2
for idx in idx_list:
    mask=mask_list[idx]
    indexs=mask_indexs[idx]

    tif.imwrite(f"{dir}/{indexs[0]}_{indexs[1]}_{indexs[2]}.tif",mask)
# %%
