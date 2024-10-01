from abc import ABC, abstractmethod
from .utils import verbose
import numpy as np

class Stopper:
    
    def __init__(self,
                 conditions,
                 verbose,
                 ):
        self.conditions = conditions
        self.verbose = verbose

    def stop(self, level):
        level.stop_info = {condition.name:condition(level)
                           for condition in self.conditions}
        self.print_stop_info(level)
    
    @verbose
    def print_stop_info(self, level):
        for item in level.stop_info.items():
            if item[1]:
                print(f"{item[0]} stopper has stopped level {level.name}.")


class Condition(ABC):

    @property
    def name(self):
        return self._name()

    @abstractmethod
    def _name(self):
        pass

    @abstractmethod
    def __call__(self,bud):
        pass

class Failure(Condition):

    def __init__(self,
                 threshold,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def _name(self):
        return 'Failure'

    def __call__(self, level):
        bud = level.trivial_bud
        return bud.seeds[0].performance >= self.threshold
    
class ThresholdConverge(Condition):

    def _name(self):
        return 'Threshold Converge'

    def __call__(self, level):
        bud = level.trivial_bud
        return bud.seeds[0].performance <= bud.parent_level.indicator.threshold
    

class NoMovement(Condition):

    def _name(self):
        return 'No Movement'

    def __call__(self, level):
        return all(np.array_equal(chain[0].array,chain[-1].array) for chain in level.chain_list)



  
