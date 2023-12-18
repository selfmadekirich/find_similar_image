from imports import *

from ImageWokerBase import ImageWorkerBase

class ImageWorkerKNN(ImageWorkerBase):
    def __init__(self,db_path):
        super().__init__(db_path)
        self.nbrs = load('nbrs.joblib')
    
    def find_n_similar(self,image:np.array,n_similar=5):
        img_vec = super().__vectorize_image__(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        _,inds = self.nbrs.kneighbors([img_vec],n_neighbors=n_similar)
        return inds.flatten()
