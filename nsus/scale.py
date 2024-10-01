from abc import ABC, abstractmethod
import numpy as np


class Scaler(ABC):

    @abstractmethod
    def compute_scale(self, seeds):
        pass

class ConstantScaler(Scaler):

    def __init__(self, scale):
        self.scale = scale

    def compute_scale(self, seeds):
        return np.array([self.scale]* seeds[0].array.shape[0])
    
class AdaptiveScaler(Scaler):
    
    def compute_scale(self, seeds):
        stds = np.std([samp.array for samp in seeds],axis=0)
        return stds
    