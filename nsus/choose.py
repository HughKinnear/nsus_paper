from abc import ABC, abstractmethod

class Chooser(ABC):

    @abstractmethod
    def choose(self, active_buds):
        pass

class DepthChooser(Chooser):

    def choose(self, active_buds):
        min_depth = min([bud.parent_level.depth for bud in active_buds])
        return [bud for bud in active_buds if bud.parent_level.depth == min_depth][0]

