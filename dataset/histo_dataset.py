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

class histo_dataset(Dataset):
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
                patch=np.zeros(shape=(self.radius*2+1,self.radius*2+1,3))
                u,d,l,r=determine_patch_boundaries(img.shape[:-1],(i,j),radius)
                # print(f"u:{u},d:{d},l:{l},r:{r}")
                patch[radius-u:radius+d,radius-l:radius+r,:]=img[i-u:i+d,j-l:j+r,:]
                self.patch_list.append(patch)
            

    def __len__(self):
        return self.img.shape[0]*self.img.shape[1]

    def __getitem__(self, idx):
        # preprocess and resize to input_shape
        patch=self.patch_list[idx]

        preprocess = v2.Compose([
        v2.Resize(size=(self.net_input_shape[0],self.net_input_shape[1])),
        #following two lines convert to tensor
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(                        # Normalize pixel values
            mean=[0.485, 0.456, 0.406],             # Precomputed ImageNet mean
            std=[0.229, 0.224, 0.225]               # Precomputed ImageNet std
        )
        ])      
        patch = preprocess(patch)
        return patch






