from anytree import findall_by_attr, PreOrderIter
from .performance_function import PerformanceFunction
from .stop import Stopper
from .level_create import LevelCreator
from .bud_create import BudCreator
from .utils import verbose
from .markov_chain import ChainData
from .level_create import Sample
from .estimate import estimate_leaf_probability


class NSuS:

    def __init__(self,
                 performance_function,
                 level_probability,
                 level_size,
                 dimension,
                 stop_conditions,
                 markov_chain,
                 partitioner,
                 seed_method,
                 allocator,
                 scaler,
                 random_state,
                 verbose,
                 chooser,
                 ):
        self.level_probability = level_probability
        self.level_size = level_size
        self.dimension = dimension
        self.stop_conditions = stop_conditions
        self.partitioner = partitioner
        self.markov_chain = markov_chain
        self.random_state = random_state
        self.allocator = allocator
        self.verbose = verbose
        self.chooser = chooser
        self.scaler = scaler
        self.initial_level = None
        self.performance_function = PerformanceFunction(performance_function)
        self.stopper = Stopper(self.stop_conditions,
                               self.verbose,
                               )
        self.bud_creator = BudCreator(seed_method,
                                      self.level_probability,
                                      self.scaler,
                                      self.verbose,
                                      )
        self.level_creator = LevelCreator(self.markov_chain,
                                          self.dimension,
                                          self.level_size,
                                          self.random_state,
                                          self.verbose,
                                          self.performance_function,
                                          self.allocator,
                                          )
        
    def run(self):
        while not self.is_stop:
            if self.is_initial:
                self.initial_level = self.level_creator.create_initial()
                level = self.initial_level
            else:
                bud = self.chooser.choose(self.options)
                level = self.level_creator.create(bud)
            self.bud_creator.create_trivial(level)
            self.stopper.stop(level)
            if not level.is_stop:
                partition_indicators = self.partitioner.partition(level)
                if len(partition_indicators) > 1:
                    self.bud_creator.create_inidicator_buds(level, partition_indicators)
        self.print_terminate()

    def bulk(self,threshold,percent_limit):
        leaf_probs = [estimate_leaf_probability(threshold,leaf) for leaf in self.leaves]
        total = sum(leaf_probs)
        leaf_percentages = [prob/total for prob in leaf_probs]
        filter_leaves = [leaf for leaf,percentage in zip(self.leaves,leaf_percentages)
                         if percentage > percent_limit]
        levels = sum([leaf.branch for leaf in filter_leaves],[])
        budget = int(self.level_size * (1-self.level_probability))
        for level in levels:
            if level.order == 1:
                continue
            markov_chain = self.level_creator.markov_chain
            chain_number = len(level.chain_list)
            chain_data = ChainData([[sample.array for sample in chain] for chain in level.chain_list])
            bud = level.parent_bud
            markov_chain.indicator = bud.indicator
            self.markov_chain.scale = bud.scale
            missing_budget = budget - level.budget
            update = missing_budget // chain_number
            markov_chain.update(chain_data, update)
            sample_chain_list = [[Sample(array=array,
                                         performance=self.performance_function(array))
                                  for array in chain]
                                 for chain in chain_data.chain_list]
            level.chain_list = sample_chain_list

    @property
    def is_initial(self):
        return self.initial_level is None

    @property
    def options(self):
        return self.initial_level.non_used_non_stop_buds
    
    @property
    def is_stop(self):
        if self.is_initial:
            return False
        return not bool(self.options)
    
    @verbose
    def print_terminate(self):
        print("No more options, algorithm terminated.")
    
    def find(self, name):
        return findall_by_attr(self.initial_level, name)[0]

    @property
    def all_levels(self):
        all_levels = list(PreOrderIter(self.initial_level))
        return sorted(all_levels, key=lambda x: x.order)

    @property
    def all_samples(self):
        return [samp for level in self.all_levels
                for samp in level.sample_list]
    
    @property
    def leaves(self):
        return self.initial_level.leaves
    