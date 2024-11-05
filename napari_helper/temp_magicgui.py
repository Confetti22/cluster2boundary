import napari
import numpy as np
from napari.layers import Image
from magicgui import magicgui

@magicgui(image={'label': 'Pick an Image'})
def my_widget(image: Image):
    print('in my_widegt')

viewer = napari.view_image(np.random.rand(64, 64), name="My Image")
viewer.window.add_dock_widget(my_widget)


if __name__ == '__main__':
    napari.run()