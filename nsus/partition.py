from abc import ABC, abstractmethod


class Partitioner(ABC):

    def __init__(self,
                 verbose):
        self.verbose = verbose

    @abstractmethod
    def partition(self, level):
        pass
    

class NoPartitioner(Partitioner):
    
    def partition(self, level):
        return [lambda x: 1]
    




