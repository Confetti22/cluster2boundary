import napari
import numpy as np
viewer=napari.Viewer()
layer1=viewer.add_image(np.zeros(shape=(9,9)),name='bee1')
layer2=viewer.add_image(np.zeros(shape=(9,9)),name='bee3')
layer3=viewer.add_image(np.zeros(shape=(9,9)),name='bee2')
layer4=viewer.add_image(np.zeros(shape=(9,9)),name='doggyi')


pattern='bee'

@viewer.mouse_double_click_callbacks.append
def my_function(layer,event):
    layer_names=[layer.name for layer  in viewer.layers if pattern in layer.name]
    for layer_name in layer_names:
        visible_attri=viewer.layers[layer_name].visible
        print(f"visible_attrii in layer:{layer_name} is {visible_attri}")

napari.run()

#%%
class C:
    def __init__(self):
        self._x = None

    def getx(self):
        return self._x
    def setx(self, value):
        self._x = value
    def delx(self):
        del self._x
    x = property(getx, setx, delx, "I'm the 'x' property.")
# %%
