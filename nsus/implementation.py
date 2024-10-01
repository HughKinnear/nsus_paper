from .nsus import NSuS
from .stop import Failure, ThresholdConverge
from .markov_chain import ModifiedMetropolis
from .partition import NoPartitioner
from .allocate import SubsetAllocator, BranchingAllocator
from .hill_valley_graph_partitioner import HillValleyGraphPartitioner
import numpy as np
from .lsvc import LSVCCreator
from .choose import DepthChooser
from .scale import ConstantScaler, AdaptiveScaler
        

class SubsetSimulation(NSuS):

    def __init__(self,
                 performance_function,
                 level_probability,
                 level_size,
                 dimension,
                 seed,
                 threshold,
                 scale,
                 verbose):

        random_state = np.random.default_rng(seed)
        stop_conditions = [
            Failure(threshold=threshold),
            ThresholdConverge(),
        ]
        chooser = DepthChooser()

        if scale is None:
            scaler = AdaptiveScaler()
        else:
            scaler = ConstantScaler(scale)
        
        markov_chain = ModifiedMetropolis(
                                          indicator=None,
                                          scale=None,
                                          random_state=random_state)
        partitioner = NoPartitioner(verbose=verbose)
        allocator = SubsetAllocator(verbose=verbose,
                                    level_probability=level_probability,
                                    level_size=level_size)
        
        seed_method = 'seeds_prob_exclusive'
    
        super().__init__(
                         performance_function=performance_function,
                         level_probability=level_probability,
                         level_size=level_size,
                         dimension=dimension,
                         stop_conditions=stop_conditions,
                         markov_chain=markov_chain,
                         partitioner=partitioner,
                         seed_method=seed_method,
                         allocator=allocator,
                         scaler=scaler,
                         random_state=random_state,
                         verbose=verbose,
                         chooser=chooser,
                         )
        
                

class HillValleyGraphNSuS(NSuS):

    def __init__(self,
                 performance_function,
                 level_probability,
                 level_size,
                 dimension,
                 seed,
                 threshold,
                 graph_size,
                 max_branches,
                 scale,
                 verbose,
                 is_cache=True):
        
        params = {'penalty': 'l1', 'C':1,'dual':'auto','max_iter':1000000}
        iterations = 100
            
        stop_conditions = [
            Failure(threshold=threshold),
            ThresholdConverge(),
        ]
        chooser = DepthChooser()
        random_state = np.random.default_rng(seed)

        if scale is None:
            scaler = AdaptiveScaler()
        else:
            scaler = ConstantScaler(scale)
        
        markov_chain = ModifiedMetropolis(
                                          indicator=None,
                                          scale=None,
                                          random_state=random_state)
               
        classifier_creator = LSVCCreator(params=params,
                                         random_state=random_state)
        
        partitioner = HillValleyGraphPartitioner(iterations=iterations,
                                            classifier_creator=classifier_creator,
                                             graph_size=graph_size,
                                             max_branches=max_branches,
                                             verbose=verbose,
                                             is_cache=is_cache,
                                             random_state=random_state)
        allocator = BranchingAllocator(verbose=verbose,
                                           level_probability=level_probability,
                                           level_size=level_size)
        
        seed_method = 'seeds_prob_exclusive'
    
        super().__init__(
                         performance_function=performance_function,
                         level_probability=level_probability,
                         level_size=level_size,
                         dimension=dimension,
                         stop_conditions=stop_conditions,
                         markov_chain=markov_chain,
                         partitioner=partitioner,
                         seed_method=seed_method,
                         allocator=allocator,
                         scaler=scaler,
                         random_state=random_state,
                         verbose=verbose,
                         chooser=chooser,
                         )
        


        
        

    



        

