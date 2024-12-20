from napari_helper.read_ims import Ims_Image
from skimage.measure import shannon_entropy
import tifffile as tif


def entropy_filter(l_thres=1.4, h_thres=100):
    def _filter(img):
        entropy=shannon_entropy(img)
        if (entropy>= l_thres) and (entropy <= h_thres):
            print(f"entrop of the roi is {entropy}")
            return True
        else:
            return False
    return _filter

save_dir="/home/confetti/mnt/data/processed/t1779/1000roi"
image_path = "/home/confetti/mnt/data/processed/t1779/t1779.ims"
level = 0
channel = 2
roi_size =(128,128,128)
amount = 1000
cnt = 1

ims_vol = Ims_Image(image_path, channel=channel)
vol_shape = ims_vol.info[level]['data_shape']

while cnt < amount:

    roi,indexs=ims_vol.get_random_roi(filter=entropy_filter(l_thres=4),roi_size=roi_size,level=0)
    file_name= f"{save_dir}/{cnt:04d}.tif"
    print(f"{file_name} has been saved ")
    tif.imwrite(file_name,roi)
    cnt = cnt +1



