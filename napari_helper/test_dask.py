import napari
import numpy as np

viewer = napari.Viewer()
mask=np.zeros(shape=(3,3,3),dtype=int)
viewer.add_labels(mask, name='mask')
napari.run()