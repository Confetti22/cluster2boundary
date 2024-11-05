
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


class SvcPredictor():

    def __init__(self,gamma=2,C=1,kernel='rbf',classweight='balanced',norm=True,config=None) -> None:   

        self.whether_norm=norm 
        self.scaler = StandardScaler()
        self.C=C # decreasing C --> more smooth (---> more regularization --> avoid overfitting--> more smooth)
        self.gamma=gamma
        self.kernel=kernel
        self.classweight=classweight
        self.config=config
        self.model = None



    #train and predict on the same input points_coords
    def findboundary(self, X, Y):
        print(f"current svc: ^_^  Gamma: {self.gamma}  Regularization: {self.C}  Kernel: {self.kernel}")

        # X = point_coords.reshape(-1,3)
        # Y = Y.flatten()  # Flatten to 1D array
        classes=np.unique(Y)

        class_weight = compute_class_weight(self.classweight, classes=classes, y=Y)
        class_weight_dict = {label: weight for label, weight in zip(classes, class_weight)}

        if self.whether_norm:
            X = self.scaler.fit_transform(X)

        self.model = SVC(C=self.C,gamma=self.gamma,kernel=self.kernel, class_weight=class_weight_dict)
        self.model.fit(X, Y)

         # Predict on the grid

        predictions = self.model.predict(X)
        predictions=predictions.reshape(self.config['roi_size'])

        return predictions

    def findboundary2d(self,X,Y,roi_size):
        print(f"current svc: ^_^  Gamma: {self.gamma}  Regularization: {self.C}  Kernel: {self.kernel}")

        classes = np.unique(Y)
        # Compute class weights based on the frequencies of each class
        class_weight = compute_class_weight(class_weight=self.classweight, classes=classes, y=Y)
        # Create a mapping of class labels to weights
        class_weight_dict = {label: weight for label, weight in zip(classes, class_weight)}
        print(f"class_weight:{class_weight_dict}")


        clf = SVC(C=self.C,gamma=self.gamma,kernel=self.kernel, \
                  class_weight=class_weight_dict,decision_function_shape='ovr')

        if self.whether_norm:
            X=self.scaler.fit_transform(X)
        clf.fit(X, Y)

        # # Predict on the grid
        # grid_points = X 
        # if self.whether_norm:
        #     grid_points=self.scaler.fit_transform(grid_points)
        # predictions = clf.predict(grid_points)

        fig, ax = plt.subplots()
    
        #plot the gt 
        ax.plot(X[Y == 1][:, 1], X[Y == 1][:, 0], 'ob', markersize=1.8)
        ax.plot(X[Y == 0][:, 1], X[Y == 0][:, 0], 'or', markersize=1.8)
        
        # Generate decision boundary
        i_interval = np.linspace(0, roi_size[0], 1000)
        j_interval = np.linspace(0, roi_size[1], 1000)
        i_g, j_g = np.meshgrid(i_interval, j_interval, indexing='ij')
        _indices = np.stack((i_g.ravel(), j_g.ravel()), axis=-1)
        
        if self.whether_norm:
            _indices = scaler.fit_transform(_indices)
            
        Z = clf.decision_function(_indices)
        Z = Z.reshape(j_g.shape)
        
        # Draw the decision boundary
        ax.contour(j_g, i_g, Z, levels=[0], linewidths=1, colors='g')
        
        # Return the Figure object
        return fig                


    def _create_meshgrid(self, ):
        # Create a grid over the volume
        roi_size=self.config['roi_size']

        # Create a grid with a reasonable number of points
        grid0 = np.linspace(0, roi_size[0]-1, num=roi_size[0])
        grid1 = np.linspace(0, roi_size[1]-1, num=roi_size[1])
        grid2 = np.linspace(0, roi_size[2]-1, num=roi_size[2])

        # Create the meshgrid
        grid_x, grid_y, grid_z = np.meshgrid(grid0, grid1, grid2)
        grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T

        return grid_points



def build_svc():

    pass

svc_model_registry = {
    "default": build_svc,
    "vit_h": build_svc,
}
