from torch.utils.data import Dataset
import numpy as np
import torch

from torchvision.transforms import v2

def determine_patch_boundaries(img_shape, center, radius):
    """
    Determine the boundaries of a patch centered at a given pixel within an image.

    Parameters:
        img_shape (tuple): Shape of the image as (height, width).
        center (tuple): Center of the patch as (row, column).
        radius (int): Radius of the patch.

    Returns:
        tuple: (u, d, l, r), where:
            u (int): Number of pixels the patch can extend upwards.
            d (int): Number of pixels the patch can extend downwards.
            l (int): Number of pixels the patch can extend leftwards.
            r (int): Number of pixels the patch can extend rightwards.
    """
    row, col = center
    height, width = img_shape

    # Calculate boundaries considering image limits
    u = min(radius, row)  # Upwards extent
    d = min(radius + 1, height - row)  # Downwards extent
    l = min(radius, col)  # Leftwards extent
    r = min(radius + 1, width - col)  # Rightwards extent

    return u, d, l, r

class visor_dataset(Dataset):
    def __init__(self,img,stride,radius,net_input_shape,num_channel):
        """
        if num_channel=3, repeat the image to 3 channel, adapt to network pretrained on natural images 
        """
        self.img=img
        self.stride=stride
        self.radius=radius
        self.patch_list=[]
        self.net_input_shape=net_input_shape
        self.num_channel=num_channel


        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                patch=np.zeros(shape=(self.radius*2+1,self.radius*2+1))
                u,d,l,r=determine_patch_boundaries(img.shape,(i,j),radius)
                # print(f"u:{u},d:{d},l:{l},r:{r}")
                patch[radius-u:radius+d,radius-l:radius+r]=img[i-u:i+d,j-l:j+r]
                self.patch_list.append(patch)
            

    def __len__(self):
        return self.img.shape[0]*self.img.shape[1]

    def __getitem__(self, idx):
        # preprocess and resize to input_shape
        patch=self.patch_list[idx]
        patch=self.preprocess(patch,num_channel=self.num_channel)
        patch=v2.Resize(size=(self.net_input_shape[0],self.net_input_shape[1]))(patch)
        
        return patch

    @staticmethod
    def preprocess(img,percentiles=[0.1,0.999],num_channel=1):
        """
        first clip the image to percentiles [0.1,0.999]
        second min_max normalize the image to [0,1]
        if num_channel=3, repeat the image to 3 channel
        """
        # input img nparray [0,65535]
        # output img tensor [0,1]
        flattened_arr = np.sort(img.flatten())
        clip_low = int(percentiles[0] * len(flattened_arr))
        clip_high = int(percentiles[1] * len(flattened_arr))-1
        # if flattened_arr[clip_high]<self.bg_thres:
        #     return None
        clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])
        min_value = np.min(clipped_arr)
        max_value = np.max(clipped_arr)
        if max_value==min_value:
            # print(f"max_vale{max_value}=min_value{min_value}\n")
            max_value=max_value+1
        filtered = clipped_arr
        img = (filtered-min_value)/(max_value-min_value)

        img = img.astype(np.float32)
        img = torch.from_numpy(img)
        # img = img.unsqueeze(0).unsqueeze(0)
        img = img.unsqueeze(0)
        img=img.repeat(num_channel,1,1)
        return img






