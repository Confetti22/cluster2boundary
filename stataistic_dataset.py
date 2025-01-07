#%%
import tifffile as tif
import numpy as np
import os

roi_size =(128,128,128)

save_dir="/home/confetti/mnt/data/processed/t1779/128roi_skip_gapped"
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
percentiles=[0.1,0.99999]
flattened_arr = np.sort(img.flatten())
clip_low = int(percentiles[0] * len(flattened_arr))
clip_high = int(percentiles[1] * len(flattened_arr))-1
print(f"{percentiles[0]} is {flattened_arr[clip_low]}")
print(f"{percentiles[1]} is {flattened_arr[clip_high]}")
print(f"min : {np.min(arr)}")
print(f"2percente: {np.percentile(arr,2)}")
print(f"98percente: {np.percentile(arr,98)}")
print(f"max: {np.max(arr)}")
print(f"mean: {np.mean(arr)}")
print(f"median: {np.median(arr)}")
#%%
import matplotlib.pyplot as plt
# Plotting the histogram
# plt.hist(arr.flatten(), bins=512, range=(0,np.percentile(arr,100) ), color='blue', alpha=0.7, edgecolor='black')
filtered_data = arr[(arr >= 2000) & (arr <= 6000)]
print(f"10 percentile {np.percentile(filtered_data,10)}")
print(f"50 percentile {np.percentile(filtered_data,50)}")
print(f"80 percentile {np.percentile(filtered_data,80)}")
print(f"95 percentile {np.percentile(filtered_data,95)}")

plt.hist(filtered_data.flatten(), bins=512, range=(2000,6000 ), color='blue', alpha=0.7, edgecolor='black')
plt.title("Histogram of Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xlim(2200,14000)
plt.show()





# %%
