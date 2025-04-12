from skimage import data
from skimage.util import img_as_float
import tifffile as tif
import numpy as np
from magicgui import magicgui,widgets
from magicgui.widgets import Container
import napari
from train_helper import get_eval_data,MLP
import torch
from scipy.ndimage import zoom
from 
import matplotlib.pyplot as plt

def load_mlp():
    ckpt_pth = "runs/test14_8192_batch4096_nview2_pos_weight_2_shuffle_every50/model_epoch_34699.pth"
    device ='cuda'
    mlp = MLP().to(device)
    mlp.eval()
    print(f"begin loading ckpt")
    ckpt= torch.load(ckpt_pth)
    mlp.load_state_dict(ckpt)
    print(f"After loading ckpt")
    return mlp

def generate_feas_map(feats,img_shape,stride):
    mlp = load_mlp()
    feats = torch.from_numpy(feats).float().to('cuda')
    encoded = mlp(feats) #N*C
    encoded = encoded.detach().cpu().numpy()
    encoded = encoded.reshape(93,93,-1) # n*n*C

    # Only zoom the first two dimensions (height and width), not channels
    zoom_factors = (stride, stride, 1)  # No zoom on channel dim
    zoomed = zoom(encoded, zoom=zoom_factors, order=1)

    # Only pad the first two dimensions (height and width)
    # padded = np.pad(zoomed, pad_width=((24, 24), (24, 24), (0, 0)), mode='constant')

    return zoomed 

import numpy as np
from magicgui import magicgui
from napari.types import LabelsData, ImageData
from napari.layers import Labels, Image
from napari.viewer import Viewer
from sklearn.preprocessing import normalize


def draw_pair_wise_cosine(X,N,title):
    # Normalize vectors
    X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)

    # Compute cosine similarity matrix
    cos_sim_matrix = X_norm @ X_norm.T  # Shape: (N, N)

    # Extract upper triangle (excluding diagonal) to avoid duplicate/self-similarity
    i_upper = np.triu_indices(N, k=1)
    pairwise_cosines = cos_sim_matrix[i_upper]

    # Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(pairwise_cosines, bins=50, color='skyblue', edgecolor='black')
    plt.title(f"Cosine Sim_{title}")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def seg_func(label_mask: np.ndarray, feature_map: np.ndarray) -> np.ndarray:
    pos_mask = label_mask == 1
    neg_mask = label_mask == 2

    if not np.any(pos_mask) or not np.any(neg_mask):
        return np.zeros(label_mask.shape, dtype=np.uint8)

    # Get feature vectors
    pos_feats = feature_map[pos_mask]
    neg_feats = feature_map[neg_mask]

    # Normalize features
    pos_feats = normalize(pos_feats, axis=1)
    neg_feats = normalize(neg_feats, axis=1)

    # Mean positive feature
    mean_pos = np.mean(pos_feats, axis=0)

    # Cosine similarity with mean_pos
    neg_cos = neg_feats @ mean_pos
    
    thres = np.max(neg_cos)
    print(f"mean of neg_cos {np.mean(neg_cos)}")
    print(f"std of neg_cos {np.std(neg_cos)}")

    # draw_pair_wise_cosine(pos_feats,pos_feats.shape[0],title='pos_pairs')
    # draw_pair_wise_cosine(neg_feats,neg_feats.shape[0],title='neg_pairs')
    mean_neg_cos = np.mean(neg_cos)
    max_neg_cos = np.max(neg_cos)
    plt.figure(figsize=(10,10))
    plt.hist(neg_cos,bins=80)
    plt.title(f'distri of cos between neg and mean_pos, max:{max_neg_cos:.4f}mean:{mean_neg_cos:.4f}')
    plt.show()
    

    # Flattened feature map
    H, W, C = feature_map.shape
    flat_feats = feature_map.reshape(-1, C)
    flat_feats = normalize(flat_feats, axis=1)

    cos_sim = flat_feats @ mean_pos
    seg_result = (cos_sim >= thres).astype(np.uint8)
    seg_result = seg_result.reshape(H, W)

    # Map to label values
    seg_label = np.zeros((H, W), dtype=np.uint8)
    seg_label[seg_result == 1] = 1  # positive
    seg_label[seg_result == 0] = 2  # negative

    return seg_label



sample_num = 2
eval_data = get_eval_data(img_no_list=[1,2,3],ncc_seed_point=False)
data_dic = eval_data[sample_num]
feats=data_dic['feats']
z_slice = data_dic['z_slice']
z_slice = z_slice[24:-24,24:-24]
img_shape = z_slice.shape
stride = 16
feature_map = generate_feas_map(feats=feats,img_shape=img_shape,stride=stride)

viewer = napari.Viewer(ndisplay=2)
viewer.add_image(z_slice,name ='img')

label_data = np.zeros((img_shape),dtype=np.uint8)
label_layer = viewer.add_labels(label_data,name ='Label')
label_layer.brush_size = 30
label_layer.mode = 'PAINT'

# --- Define separate buttons ---
seg_button = widgets.PushButton(text="Seg")
clear_button = widgets.PushButton(text="Clear")

# --- Seg button action ---
@seg_button.clicked.connect
def run_seg():
    seg_out_name ='seg_out'
    label_data = label_layer.data
    seg_result = seg_func(label_data, feature_map)

    #add seg_out data
    
    if viewer.layers

    else:
        viewer.add_labels(seg_result, name=seg_out_name)

    viewer.layers.selection = [label_layer]  # Keep selected
# --- Clear button action ---

@clear_button.clicked.connect
def clear_labels():
    label_layer.data[:] = 0
    viewer.layers.selection = [label_layer]  # Keep selected

# --- Combine buttons into a container widget ---
control_panel = Container(widgets=[seg_button, clear_button])

# Add widget to napari
viewer.window.add_dock_widget(control_panel, area='right')
napari.run()

