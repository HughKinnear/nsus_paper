from anytree import AnyNode, PreOrderIter
import numpy as np


class Level(AnyNode):

    def __init__(self,
                 parent_bud,
                 chain_list,
                 *args,
                 **kwargs
                 ):
        super().__init__(parent=parent_bud.parent_level,
                         *args,
                         **kwargs)

        self.parent_bud = parent_bud
        self.chain_list = chain_list
        self.empty_indicators = []
        self.trivial_bud = None
        self._buds = []
        self.extra_info = {}
        self.stop_info = None
        self.name = str(self.root_size)
        self.indicator = self.parent_bud.indicator
        self.parent_bud.child_level = self
    
    @property
    def buds(self):
        if bool(self._buds):
            return self._buds
        return [self.trivial_bud]
    
    @property
    def root_size(self):
        return self.root.size
    
    @property
    def order(self):
        return int(self.name)

    @property
    def sorted_list(self):
        return sorted(self.sample_list, key=lambda x: x.performance)

    @property
    def unique_list(self):
        return [self.sample_list[ind]
                for ind in np.unique([samp.array for samp in self.sample_list],
                                     axis=0,
                                     return_index=True)[1]]
    
    @property
    def level_size(self):
        return len(self.sample_list)

    @property
    def branch(self):
        return list(self.ancestors) + [self]
    
    @property
    def sample_list(self):
        return [sample
                for chain in self.chain_list
                for sample in chain]
    
    @property
    def budget(self):
        if self.order == 1:
            return self.level_size
        return self.level_size - len(self.chain_list)
    
    @property
    def is_stop(self):
        if self.stop_info is None:
            return False
        return any(self.stop_info.values())
    
    @property
    def non_used_buds(self):
        non_used_buds = []
        for level in PreOrderIter(self.root):
            for bud in level.buds:
                if not bud.is_used:
                    non_used_buds.append(bud)
        return non_used_buds
    
    @property
    def non_used_non_stop_buds(self):
        return [bud
                for bud in self.non_used_buds
                if not bud.parent_level.is_stop]
    
    @property
    def acceptance_rate(self):
        jumps = 0
        succ_jumps = 0
        for chain in self.chain_list:
            for samp_a, samp_b in zip(chain[:-1],chain[1:]):
                jumps += 1
                if not np.array_equal(samp_a.array,samp_b.array):
                    succ_jumps += 1
        return succ_jumps/jumps
    

        
    
    