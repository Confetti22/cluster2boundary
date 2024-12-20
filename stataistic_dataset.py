#%%
import tifffile as tif
import numpy as np
import os

roi_size =(128,128,128)

save_dir="/home/confetti/mnt/data/processed/t1779/100roi"
files=[os.path.join(save_dir,fname) for fname in os.listdir(save_dir) if fname.endswith('.tif')]
number = len(files)
list=[]

for file in files:
    img = tif.imread(file)
    img = np.ravel(img)
    list.append(img)
#%%
arr = np.array(list)
#%%
print(f"min : {np.min(arr)}")
print(f"max: {np.max(arr)}")
#%%
import matplotlib.pyplot as plt
# Plotting the histogram
plt.hist(arr.flatten(), bins=512, range=(0, max), color='blue', alpha=0.7, edgecolor='black')
plt.title("Histogram of Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()





# %%
