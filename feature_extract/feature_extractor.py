from dataset.visor_dataset import visor_dataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision
import pickle
from typing import List
#if num_channel=3, repeat the image to 3 channel, adapt to network pretrained on natural images 
num_channel=3


def get_encoder(device,model_name='resnet18'):
    if model_name == 'resnet18':
        model=torchvision.models.resnet18(weights='ResNet18_Weights.IMAGENET1K_V1').to(device)
    if model_name == "inceptionv3":
        model=torchvision.models.inception_v3(weights="Inception_V3_Weights.IMAGENET1K_V1")
    
    model.eval()
    return model

def get_feature_list(device,encoder,test_loader,extract_layer_name,feats_dim=512,save=False,save_path=None)->np.ndarray:
    """
    encoder inference on a single input 2d-image
    
    input(numpy)--> test_dataset
    collect feats during inference
    return the feats as shape of N*n_dim

    """
    print(f"device is {device}")


    # a dict to store the activations
    activation = {}
    def getActivation(name):
        # the hook signature
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    #register the forward hook at layer"layer_name"
    hook1 = getattr(encoder,extract_layer_name).register_forward_hook(getActivation(extract_layer_name))

    feats_list=[]
    for i, imgs in enumerate(tqdm(test_loader,desc="extracting features")):
        outs=encoder(imgs.to(device))
        feats_list.append(activation[extract_layer_name].cpu().detach().numpy())
    
    #detach the hook
    hook1.remove()
    feats_array = np.concatenate([arr.reshape(-1, feats_dim) for arr in feats_list], axis=0)

    if save:
        if not save_path :
            save_path='feats.pkl'
        
        with open(save_path, 'wb') as file:
            pickle.dump(feats_array, file)
        

    
    return feats_array