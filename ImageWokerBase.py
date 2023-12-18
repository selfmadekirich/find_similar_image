from imports import *

class ImageWorkerBase():
    def __init__(self,db_path:str):
        self.__db_load__(db_path)
        self.sift = cv.SIFT_create()
        self.kmeans = load('kmeans.joblib')
    
    def __db_load__(self,db_path:str):
        self.db = pd.read_csv(db_path,delimiter='\t')
        self.db['encoding_vector'] = self.db['encoding_vector'].apply(lambda x: eval(x))
    
    def __vectorize_image__(self,image:np.array,n_clusters=512):
        _,des = self.sift.detectAndCompute(image,None)
        classes = self.kmeans.predict(des)
        hist = np.zeros(n_clusters)
        for clss in classes:
            hist[clss] += 1

        hist /= len(classes)
        return hist

    def get_image(self,ind:int):
        if ind < 0 or ind > len(self.db):
            return np.zeros((300,300))
        return cv.imread(self.db.loc[ind]['filepath'])
        
    def find_n_similar(self,image:np.array,n_similar=5):
        return 'BASE_CLASS'
