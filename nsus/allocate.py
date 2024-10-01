from abc import ABC, abstractmethod


class Allocator(ABC):
    
    def __init__(self,
                 verbose,
                 level_probability,
                 level_size):
        self.verbose = verbose
        self.level_probability = level_probability
        self.level_size = level_size
        self.budget = int((1-self.level_probability) * self.level_size)
    
    @abstractmethod
    def allocate(self, bud):
        pass


class SubsetAllocator(Allocator):
    
    def allocate(self, bud):
        return int(self.level_probability ** -1)
    

class BranchingAllocator(Allocator):
    
    def allocate(self, bud):
        split_budget = self.budget / len(bud.parent_level.non_used_buds)
        return int(split_budget / len(bud.seeds)) + 1
    














