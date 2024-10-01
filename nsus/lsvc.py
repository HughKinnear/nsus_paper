from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import numpy as np
from sklearn.decomposition import PCA


class BarebonesPipeline(Pipeline):

    def predict_single(self, x):
        for _, transformer in self.steps[:-1]:
            x = transformer.transform_single(x)
        return self.steps[-1][1].predict_single(x)


class BarebonesStandardScaler(StandardScaler):

    def transform_single(self, x):
        return (x - self.mean_) / self.var_ ** .5


class BarebonesPCA(PCA):

    def transform_single(self, x):
        return np.dot(x - self.mean_, self.components_.T)
  
    
class BarebonesLinearSVC(LinearSVC):

    def predict_single(self, x):

        scores = np.dot(self.coef_, x) + self.intercept_

        if len(scores) == 1:
            return self.classes_[int((scores > 0)[0])]
        else:
            return self.classes_[np.argmax(scores)]


class PipeLSVC:

    def __init__(self, bb_lsvc, bb_ss, bb_pca):
        self.bb_lsvc = bb_lsvc
        self.bb_ss = bb_ss
        self.bb_pca = bb_pca
        
    def fit(self, X, y):
        X = self.bb_ss.transform(X)
        X = self.bb_pca.transform(X)
        self.bb_lsvc.fit(X, y)

    def predict_single(self, x):
        x = self.bb_ss.transform_single(x)
        x = self.bb_pca.transform_single(x)
        return self.bb_lsvc.predict_single(x)
    
    def predict(self, X):
        X = self.bb_ss.transform(X)
        X = self.bb_pca.transform(X)
        return self.bb_lsvc.predict(X)
    

class LSVCCreator:

    def __init__(self,params,random_state):
        self.params = params
        self.params['random_state'] = random_state.integers(0, 1000000)

    def create(self,level):
        level_array = np.array([sample.array for sample in level.sample_list])
        bb_ss = BarebonesStandardScaler()
        bb_ss.fit(level_array)
        ss_array = bb_ss.transform(level_array)
        bb_pca = BarebonesPCA()
        bb_pca.fit(ss_array)
        bb_lsvc = BarebonesLinearSVC(**self.params)
        return PipeLSVC(bb_lsvc,bb_ss,bb_pca)


        
    



    


    





    



