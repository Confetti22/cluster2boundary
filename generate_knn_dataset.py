from napari_helper.read_ims import Ims_Image
from skimage.measure import shannon_entropy
import tifffile as tif
import numpy as np
from scipy.ndimage import zoom
import os
import math
import pickle


#first check the properity of entropy filter with thres 4 for cube of size 96 dataset (is appropriate)
#second write the code that generate knn dataset and it's distance-weight table
#will genereate (K+1) datasets: folder, folder1, folder2, ...folderK,  
#for ths ith img in folder, it's knn are the ith img in folder1 to folderk
#the distance-weight table is stored in the folder

def entropy_filter(l_thres=1.4, h_thres=100):
    def _filter(img):
        entropy=shannon_entropy(img)
        if (entropy>= l_thres) and (entropy <= h_thres):
            # print(f"entrop of the roi is {entropy}")
            return True
        else:
            return False
    return _filter

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

MASK = False
K = 3
R = 40 #sample radius for neighbour roi, only roi within this range can be sampled 
sigma = 40 #control the decrease speed of gaussian 
save_dir="/home/confetti/data/t1779/128_3nn"
image_path = "/home/confetti/mnt/data/processed/t1779/t1779.ims"
# shape is (13568, 10048, 14016)
sample_range = [[4000,7000],[0,5000],[0,7000]] 
#sample range here is to confine the dataset in a smaller region, around 1/8 of the origin brain, 
#so that total amount can be smaller to contain enough type of region
amount = 4096 

level = 0
channel = 2
roi_size =(96,96,96)
zoom_factor = 25/1
cnt = 1

ims_vol = Ims_Image(image_path, channel=channel)
vol_shape = ims_vol.info[level]['data_shape']

#build folders for knn datasets
dir_list = [] #save all the folder path for each neighbour

folder_name = os.path.basename(save_dir)
dir_name = os.path.dirname(save_dir)
os.makedirs(save_dir,exist_ok=True)

dir_list.append(save_dir)
for i in range(K):
    path =  f"{dir_name}/{folder_name}_{i+1}"
    dir_list.append(path)
    os.makedirs(path,exist_ok=True)


total_weights = np.zeros(shape=(amount,K))
while cnt < amount +1:

    #generate primary roi
    roi,start_indexs=ims_vol.get_random_roi(filter=entropy_filter(l_thres=4),roi_size=roi_size,level=0,skip_gap= True,sample_range=sample_range)
    file_name = f"{dir_list[0]}/{cnt:04d}.tif"
    tif.imwrite(file_name,roi)

    one_weights = [] 
    #randomly sample K rois within the range of R and compute weight based on euclidean distance
    for i in range(K):
        neighbour,l2_dist= ims_vol.sample_within_range(center=start_indexs, radius=R,filter=entropy_filter(l_thres=4), roi_size=roi_size, level=0)

        #save neighbour roi
        file_name = f"{dir_list[i+1]}/{cnt:04d}.tif"
        tif.imwrite(file_name,neighbour)
        #compute neighbout weight
        weight = math.exp( -l2_dist**2/(2.0*sigma*sigma))
        one_weights.append(weight)


    total_weights[cnt-1]=np.array(one_weights)
     
    print(f"{file_name} and it's neighbours has been saved ")
    print(f"weights are {one_weights}")
    cnt = cnt +1

#record weights 
with open(f'{dir_list[0]}/weights.pkl', 'wb') as f:
    pickle.dump(np.array(total_weights), f)



