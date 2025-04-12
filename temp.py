import tifffile as tif
import os 
import napari
import pickle
K = 3 
save_dir="/home/confetti/data/t1779/128_3nn"
dir_list = []

weights_pth = "/home/confetti/data/t1779/128_3nn/weights.pkl"
with open(weights_pth,'rb') as f:
    weights_table = pickle.load(f)
print(weights_table)
print(type(weights_table))
print(weights_table.shape)

indexes = [3621, 3942,  527,  848, 3009]
print(weights_table[indexes])
exit(0)

folder_name = os.path.basename(save_dir)
dir_name = os.path.dirname(save_dir)
dir_list.append(save_dir)
for folder_idx in range(K+1):
    path =  f"{dir_name}/{folder_name}_{folder_idx+1}"
    dir_list.append(path)

viewer = napari.Viewer(ndisplay=3)
for idx in range(7):
    for folder_idx in range(K+1):
        file_name = f"{dir_list[folder_idx]}/{idx+1:04d}.tif"
        img = tif.imread(file_name)
        viewer.add_image(img, name=f'{folder_idx+1}_{idx}th' )

napari.run()

