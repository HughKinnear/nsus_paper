from copy import deepcopy

class PerformanceFunction:

    def __init__(self, performance_function):
        self.non_cache_performance_function = deepcopy(performance_function)
        self.cache  = {}

    def __call__(self, x):
        tuple_x = tuple(x)
        try:
            result = self.cache[tuple_x]
        except KeyError:
            self.cache[tuple_x] = result = self.non_cache_performance_function(tuple_x)
        return result

    @property
    def eval_count(self):
        return len(self.cache)
    
        
        
    
    




