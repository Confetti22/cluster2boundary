#%%
import numpy as np 
import matplotlib.pyplot as plt 

import tifffile as tif
import os
roi_size=(24,32)


#%%
from scipy.ndimage import center_of_mass

label_lst=[]
mask_dir="/home/confetti/e5_workspace/cluster2boudnary/new_t11/test_data/test_mask_rois_2classes"
mask_list=[tif.imread(f"{mask_dir}/{filename}") for filename in os.listdir(mask_dir) ]

#visualize the masks
for idx, mask in enumerate(mask_list):
    mask=mask.squeeze()
    mask=mask[0:roi_size[0],0:roi_size[1]]
    label_lst.append(mask)
    labels, counts = np.unique(mask, return_counts=True)
    print(f"Labels: {labels}")
    print(f"Counts: {counts}")
    
    plt.figure(figsize=(6, 6))
    plt.imshow(mask)
    plt.title(f'mask idx is {idx}')
    
    # Calculate the center of each labeled region and annotate it
    for label  in labels:
        center = center_of_mass(mask == label)
        plt.text(center[1], center[0], str(label), color='red', fontsize=12, 
                 ha='center', va='center', fontweight='bold')
    
    plt.show()
#%%
################test of a way to smooth interpolate############
label=label_lst[4]
# label=np.array([[1,0,0],
#        [1,1,0],
#        [1,1,1]])

print(f"shape of origin mask {label.shape}")
plt.figure()
plt.imshow(label)
plt.title("origin mask")

import numpy as np
from scipy.ndimage import distance_transform_edt, zoom
from skimage.segmentation import watershed

def upsample_mask(mask1, scale_factor):
    # Get unique labels
    unique_labels = np.unique(mask1)
    
    # Prepare the high-resolution mask
    high_res_shape = (np.array(mask1.shape) * np.array(scale_factor)).astype(int)
    high_res_mask = np.zeros(high_res_shape, dtype=np.int32)
    
    for label in unique_labels:
        if label == 0:  # Skip background
            continue
        # Create binary mask for the current label
        binary_mask = (mask1 == label).astype(np.float32)
        
        # Upsample the binary mask using zoom
        upsampled_binary = zoom(binary_mask, scale_factor, order=1)

 
        # Use watershed segmentation to refine boundaries
        markers = (upsampled_binary > 0.5).astype(np.int32)
        # distance = distance_transform_edt(upsampled_binary > 0.5)
        # refined_region = watershed(-distance, markers, mask=(upsampled_binary > 0))
        # high_res_mask[refined_region > 0] = label
        high_res_mask[markers > 0] = label
    
    return high_res_mask

high_res_mask=upsample_mask(label,4)
plt.figure()
plt.imshow(high_res_mask)
plt.title("high_res_mask")
direct_zoomed_mask=zoom(label,zoom=4,order=3)
plt.figure()
plt.imshow(direct_zoomed_mask)
plt.title("direct zoomed_mask")


################end  of a way to smooth interpolate############


#%%
label=label_lst[4]
y_indices, x_indices = np.indices(roi_size)
#%%
indexs = np.stack((y_indices.ravel(), x_indices.ravel()), axis=-1)
print(indexs.shape)
print(label.shape)
#%%

def visualize_label_img(label,comment=""):
    plt.figure()
    plt.plot(indexs[label==0][:,1],indexs[label==0][:,0],'or',markersize=1.8)
    plt.plot(indexs[label==1][:,1],indexs[label==1][:,0],'ob',markersize=1.8)
    print(f"label 0 : red \t label 1 : blue\n")
    plt.title(comment)

def annoted_by_circle_center(ori_label,cc_lst,r=3, target_label=0):
    """
    generate circle labels with radius=r, circle center at cc_lst
    indics of circle center are given by (y_idx, x_dix)
    """
    r=float(r)
    modified=ori_label.copy()
    Y,X=ori_label.shape
    yy,xx =np.meshgrid(  np.arange(Y), np.arange(X), indexing='ij' )
    for (y,x) in cc_lst:
        distance = np.sqrt((yy - y)**2 + (xx - x)**2)
        circle_mask=distance <= r
        modified[circle_mask] = target_label
    return modified 
visualize_label_img(label.ravel(),comment='ori_label')
# modified_label=annoted_by_circle_center(label, [(0,0)],r=4,target_label=1)
modified_label=annoted_by_circle_center(label, [(17,8),(15,11),(13,15)],r=3,target_label=1)

visualize_label_img(modified_label.ravel(),comment='modified_label')

#%%
from napari_helper._build_svc import SvcPredictor


label_flatted=label.ravel()
label_flatted=modified_label.ravel()

C=0.5
gamma=1
kernel='rbf'
classweight=None 
norm=True

"""class_weight
dict, “balanced” or None
If “balanced”, class weights will be given by n_samples / (n_classes * np.bincount(y)). 
If a dictionary is given, keys are classes and values are corresponding class weights. 
If None is given, the class weights will be uniform.
"""

clf=SvcPredictor(gamma=gamma,C=C,kernel=kernel,classweight=classweight,norm=norm)
fig=clf.findboundary2d(X=indexs,Y=modified_label.ravel(),roi_size=roi_size)
fig.show()



#%%
# grid search for best C and gamma
# Parameters to try
# c_range = [1e-2, 1, 1e2]
# gamma_range = [1e-1, 1, 1e1]
c_range=np.logspace(-3,2,5)
gamma_range=np.logspace(-3,0,5)

# Setting up the grid
fig, axes = plt.subplots(5, 5, figsize=(40, 40))
fig.suptitle("SVM Decision Boundaries for Different C and Gamma Values", fontsize=16)

import time

start_time=time.time()
# Iterate over C and gamma values to generate plots
for i, C in enumerate(c_range):
    for j, gamma in enumerate(gamma_range):
        # Create an SVM model with current C and gamma
        clf = SvcPredictor(gamma=gamma, C=C, kernel='rbf', classweight=None, norm=False)
        
        # Find and plot the decision boundary
        boundary_fig = clf.findboundary2d(X=indexs, Y=modified_label.ravel(), roi_size=roi_size)
        
        # Embed the generated figure into the axes
        ax = axes[i, j]
        boundary_fig.canvas.draw()  # Draw the figure to access its content
        ax.imshow(boundary_fig.canvas.buffer_rgba(), origin='upper')
        ax.axis('off')  # Turn off axes

        # Set the title for each subplot
        ax.set_title(f"C={C:.4f}, Gamma={gamma:.4f}", fontsize=10)
# Adjust the space between subplots
end_time=time.time()
print(f"execution time:{(end_time-start_time):.2f} second")
plt.subplots_adjust(wspace=0.1, hspace=0.1)  

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


#%%
# Compute class weights based on the frequencies of each class
# class_weight = compute_class_weight('balanced', classes=classes, y=label_flatted)
# # Create a mapping of class label_flatted to weights
# class_weight_dict = {label: weight for label, weight in zip(classes, class_weight)}
# print(f"class_weight:{class_weight_dict}")

# Initialize and fit the SVM model
# model = SVC(C=regularization,gamma=gamma,kernel='rbf', class_weight=class_weight_dict)

# model = SVC(C=C,gamma=gamma,kernel='rbf',decision_function_shape='ovr')
# model.fit(indexs, label_flatted)

# # Create a meshgrid for decision boundary visualization
# grid_points = indexs 

# # Predict on the grid
# predictions = model.predict(grid_points)

# label_flatted=predictions
# plt.subplot()
# plt.plot(indexs[label_flatted==0][:,1],indexs[label_flatted==0][:,0],'*r',markersize=1.8)
# plt.plot(indexs[label_flatted==1][:,1],indexs[label_flatted==1][:,0],'*b',markersize=1.8)
# i_interval = np.linspace(0, 24, 1000)
# j_interval = np.linspace(0, 32, 1000)
# i_g, j_g = np.meshgrid(i_interval,j_interval,indexing='ij')
# _indices = np.stack((i_g.ravel(), j_g.ravel()),axis=-1)
# Z = model.decision_function(_indices)
# Z=Z.reshape(j_g.shape)
# plt.contour(j_g, i_g, Z, levels=[0], linewidths=1, colors='g')




# %%
import numpy as np
nx,ny=3,2
x=np.linspace(0,1,nx)
y=np.linspace(0,1,ny)
xv,yv=np.meshgrid(x,y)
print(xv)
print(yv)
xv,yv=np.meshgrid(x,y,indexing='ij')
print(xv)
print(yv)

#%%
import numpy as np
arr=np.array(
            [
            [[1,2,3,4],
            [4,5,6,7]],
            [[11,12,13,14],
            [14,15,16,17]],   
            [[21,22,23,24],
            [24,25,26,27]]
            ]
           )
print(arr)
print(arr.shape)
f_arr=arr.ravel()
fake_arr=f_arr.reshape(arr.shape)
print(f"fake_arr {fake_arr}")
#%%
ret1,ret2=np.indices((2,3))
print(f"ret1 {ret1}")
print(f"ret2 {ret2}")


# %%
