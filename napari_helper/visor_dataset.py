from torch.utils.data import Dataset
import numpy as np

from torchvision.transforms import v2
from utilitis import visor_preprocess,determine_patch_boundaries


class visor_dataset(Dataset):
    def __init__(self,img,stride,radius,net_input_shape):
        self.img=img
        self.stride=stride
        self.radius=radius
        self.patch_list=[]
        self.net_input_shape=net_input_shape


        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                patch=np.zeros(shape=(self.radius*2+1,self.radius*2+1))
                u,d,l,r=determine_patch_boundaries(img.shape,(i,j),radius)
                print(f"u:{u},d:{d},l:{l},r:{r}")
                patch[radius-u:radius+d,radius-l:radius+r]=img[i-u:i+d,j-l:j+r]
                self.patch_list.append(patch)
            

    def __len__(self):
        return self.img.shape[0]*self.img.shape[1]

    def __getitem__(self, idx):
        # preprocess and resize to input_shape
        patch=self.patch_list[idx]
        patch=visor_preprocess(patch)
        patch=v2.Resize(size=(self.net_input_shape[0],self.net_input_shape[1]))(patch)
        
        return patch






