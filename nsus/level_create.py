from scipy.stats import multivariate_normal
import numpy as np
from dataclasses import dataclass
from .markov_chain import ChainData
from .utils import verbose
from .level import Level
from .indicator import Indicator
from .bud_create import Bud


@dataclass
class Sample:
    array: ...
    performance: ...


class LevelCreator:
    
    def __init__(self,
                 markov_chain,
                 dimension,
                 level_size,
                 random_state,
                 verbose,
                 performance_function,
                 allocator,
                 ):
        self.markov_chain = markov_chain
        self.dimension = dimension
        self.level_size = level_size
        self.random_state = random_state
        self.verbose = verbose
        self.performance_function = performance_function
        self.allocator = allocator

    def create_initial(self):
        array = multivariate_normal.rvs(mean=np.zeros(self.dimension),
                                        cov=np.identity(self.dimension),
                                        size=self.level_size,
                                        random_state=self.random_state)
        chain_list = [[Sample(array=arrayi,
                              performance=self.performance_function(arrayi))]
                      for arrayi in array]
        initial_indicator = Indicator(threshold=-np.inf,
                                      partition_indicator=lambda x: 1,
                                      performance_function=self.performance_function)
        initial_bud = Bud(indicator=initial_indicator)
        initial_level = Level(chain_list=chain_list,
                              parent_bud=initial_bud)
        self.print_create_level(initial_level)
        return initial_level

    def create(self,bud):
        chain_data = ChainData([[seed.array] for seed in bud.seeds])
        self.markov_chain.indicator = bud.indicator
        chain_length = self.allocator.allocate(bud)
        self.markov_chain.scale = bud.scale
        self.markov_chain.update(chain_data,
                                 chain_length - 1,
                                 )
        sample_chain_list = [[Sample(array=array,
                                         performance=self.performance_function(array))
                                  for array in chain]
                                 for chain in chain_data.chain_list]
        level = Level(chain_list=sample_chain_list,
                      parent_bud=bud)
        self.print_create_level(level)
        return level

    @verbose
    def print_create_level(self, level):
        print(f"Level {level.name} created.")
            



    

    


