from numpy.core.multiarray import array as array
from imports import *

from ImageWokerBase import ImageWorkerBase

class ImageWorkerKNN(ImageWorkerBase):
    def __init__(self,db_path,use_clip=False):
        super().__init__(db_path)
        self.use_clip = use_clip
        self.nbrs = load('nbrs.joblib')

        if self.use_clip:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
    
    def __vectorize_image__(self, image: np.array, n_clusters=512):
        if not self.use_clip:
            return super().__vectorize_image__(image, n_clusters)

        image_input = self.preprocess(Image.fromarray(image.astype(np.uint8))).unsqueeze(0).to(self.device)
        return self.model.encode_image(image_input).cpu().detach().numpy().squeeze().astype(np.float64)
    
    def find_n_similar(self,image:np.array,n_similar=5):
        img_vec = super().__vectorize_image__(cv.cvtColor(image, cv.COLOR_BGR2GRAY))
        _,inds = self.nbrs.kneighbors([img_vec],n_neighbors=n_similar)
        return inds.flatten()
