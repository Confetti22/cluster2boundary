import numpy as np
from confettii.ncc_helper import get_ncc_point_to_whole,crop_2d_image
from feature_extract.feature_extractor import get_feature_list, get_encoder
from dataset.histo_dataset import histo_dataset
from torch.utils.data import DataLoader
from PIL import Image
import pickle
from torchsummary import summary
import tifffile as tif
import torch


model_name='resnet18'
device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"device{device}")
# encoder=get_encoder(device,model_name='inceptionv3')
encoder=get_encoder(device,model_name=model_name)
summary(encoder,(3,100,100))


stride=1
radius=50 #radius of patch, is determine be the resolution of image, 2*radius should be around 100 voxel_size
network_input_shape=(100,100)
num_channel=3 # for using pretrain model on natural images

img_pth="/share/home/shiqiz/data/mousebrainatlas/histology/102117913_d0.png"
# img_pth="/home/confetti/data/mousebrainatlas/histology/102117913_d0.png"
img=Image.open(img_pth)
img=np.array(img)

roi_size=600
roi=crop_2d_image(img,ori=(1660,2880),range=(roi_size,roi_size))

#cannot direct downsample to 1/4, will lose mophlogy info
# sub_roi=roi[::4,::4,:]
# roi=crop_2d_image(img,ori=(1660,2880),range=(roi_size,roi_size))


#prepare dataset for inference and extract the feats
test_dataset=histo_dataset(roi,stride,radius,net_input_shape=network_input_shape,num_channel=num_channel)
test_loader=DataLoader(test_dataset,batch_size=256,shuffle=False)
print(f"begin to extrac feats")
import time
start=time.time()
feats_map=get_feature_list(device,encoder,test_loader,extract_layer_name='avgpool',save=True,
                           save_path=f'feats_roi{network_input_shape[0]}_{model_name}.pkl')
end=time.time()
print(f"total inference time :{end-start}")


