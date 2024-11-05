import tifffile as tif
from dask.distributed import Client
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utilitis import time_execution
import dask
from dask import delayed
import dask.array as da
from dask.diagnostics import ProgressBar
from scipy.ndimage import zoom

def get_start_point_list(img):
    """
    start from four corners
    """
    start_list = []
    for i in range(img.shape[1]):
        start_list.append((0, i))
        start_list.append((img.shape[0] - 1, i))
    for i in range(img.shape[0]):
        start_list.append((i, 0))
        start_list.append((i, img.shape[1] - 1))
    return start_list


def get_neibor(cur, img):
    displacement = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    neibor_list = []
    for delta in displacement:
        if 0 <= cur[0] + delta[0] < img.shape[0] \
                and 0 <= cur[1] + delta[1] < img.shape[1]:
            neibor_list.append((cur[0] + delta[0], cur[1] + delta[1]))
    return neibor_list

def is_in_brain(cur, img,thres):
    if img[cur[0], cur[1]] >= thres:
        return True
    else:
        return False


def bfs(img,thres):
    queue = []
    visited = np.zeros(shape=img.shape, dtype=bool)
    start_list = get_start_point_list(img)
    for st in start_list:
        queue.append(st)
    while len(queue) > 0:
        cur = queue[0]
        queue.pop(0)
        if visited[cur[0], cur[1]]:
            continue
        if is_in_brain(cur, img,thres):
            continue
        visited[cur[0], cur[1]] = True
        neighbor_list = get_neibor(cur, img)
        for neighbor in neighbor_list:
            queue.append(neighbor)
    visited = 1 - visited
    return visited

@time_execution     
def get_brain_mask(vol,thres=200):
    mask = np.zeros(shape=vol.shape, dtype=np.uint8)
    for i, img in enumerate(vol):
        mask[i] = bfs(img,thres)
    return mask

@time_execution     
def get_brain_mask_dd(vol, thres=200):
    # Delayed function for parallel execution
    def process_slices(slice_chunk):
        return np.stack([bfs(img, thres) for img in slice_chunk])

    chunk_size = 10  # Number of slices per task
    vol_chunks = [vol[i:i+chunk_size] for i in range(0, len(vol), chunk_size)]
    lazy_results = [delayed(process_slices)(chunk) for chunk in vol_chunks]

    # Compute all slices in parallel and stack the results
    mask = dask.compute(*lazy_results)

    # Stack the results into a 3D mask
    return np.stack(mask)

def process_slice(img, thres=200):
    img=img.squeeze()
    mask=bfs(img, thres)
    mask=mask.reshape(1, *mask.shape)
    return mask

def process_small_vol(vol, thres=200):
    mask = np.zeros(shape=vol.shape, dtype=np.uint8)
    for i, img in enumerate(vol):
        img=img.squeeze()
        slice_mask=bfs(img, thres)
        mask[i] = slice_mask.reshape(1, *slice_mask.shape)
    return mask

@time_execution
def get_brain_mask_da(vol, thres=200,z_granu=2):
    vol_da = da.from_array(vol, chunks=(z_granu, vol.shape[1], vol.shape[2]))  # Chunk along the first dimension (z-axis)

    # Map the process_slice function to each chunk
    mask_da = vol_da.map_blocks(lambda vol: process_small_vol(vol, thres), dtype=np.uint8)

    # Compute the final mask in parallel
    mask = mask_da.compute()
    return mask

def filter_based_on_low_reso(img,mask):
    #rescale the mask to the target shape of img
    target_shape=img.shape
    source_shape=mask.shape
    zoom_factors=[t/s for t, s in zip(target_shape,source_shape)]
    target_mask=zoom(mask,zoom_factors,order=3)
    target_mask.astype(bool)
    print("target_mask geneeared")

    #apply the target_mask to img
    img[target_mask==0]=0
    return img

def generate_mask(pth):
    pth='/share/home/shiqiz/workspace/preprocess_visor/data/r32_c561.tif'
    fname=Path(pth).stem
    dir_name=Path(pth).parent
    vol=tif.imread(pth)
    brain_mask=get_brain_mask(vol,thres=200)
    tif.imwrite('r32_brain_mask.tif',brain_mask)


def appy_mask_to_all():
    channels=['405','488','561','640']
    dir_name="/share/home/shiqiz/workspace/preprocess_visor/data"
    mask=tif.imread("/share/home/shiqiz/workspace/preprocess_visor/data/r32_brain_mask.tif")

    for channel in channels:
        fname=f'r32_c{channel}.tif'
        pth=os.path.join(dir_name,fname)
        img=tif.imread(pth)
        print("img loaded")
        img[mask==0]=0
        tif.imwrite(f"{dir_name}/{fname}_filtered.tif",img)

# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster()
# client = Client(cluster)

if __name__ == '__main__':

    home_dir = os.getenv("HOME")
    path="/Users/cottonfisher/workspace/cluster2boudnary/data/r32_c488.tif"

    fname=Path(path).stem
    dirname=Path(path).parent
    volome=tif.imread(path)
    ranges=[[0,440],[35,245],[50,370]]
    volome=volome[ranges[0][0]:ranges[0][1],ranges[1][0]:ranges[1][1],ranges[2][0]:ranges[2][1]]

    small_vol=volome[0:50,:,:]

    dask.config.set(scheduler='processes') 
    # client = Client(timeout="5000s")
    # print(client.dashboard_link)
    brain_mask=get_brain_mask_da(small_vol,thres=200,z_granu=2)
    
