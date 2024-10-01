class Indicator:

    def __init__(self, threshold, partition_indicator, performance_function):
        self.threshold = threshold
        self.partition_indicator = partition_indicator
        self.performance_function = performance_function
    
    def threshold_indicator(self, x):
        return int(self.performance_function(x)
                   >= self.threshold)
    
    def indicator(self, x):
        if self.partition_indicator(x) == 1:
            if self.threshold_indicator(x) == 1:
                return 1
        return 0
    
    def __call__(self, x):
        if isinstance(x, list):
            return [self.indicator(sample.array) for sample in x]
        else:
            return self.indicator(x)
    
    @staticmethod
    def indicator_combine(indicator_a, indicator_b):
        def indicator(x):
            return indicator_a(x) * indicator_b(x)
        return indicator
    
    def child_indicator(self,threshold,partition_indicator):
        combined_partition_indicator = self.indicator_combine(self.partition_indicator,
                                                              partition_indicator)
        return Indicator(threshold,
                         combined_partition_indicator,
                         self.performance_function)
    
    def new_threshold(self, threshold):
        return Indicator(threshold,
                         self.partition_indicator,
                         self.performance_function)