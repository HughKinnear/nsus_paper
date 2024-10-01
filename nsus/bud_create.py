from .utils import verbose


class Bud:

    def __init__(self,
                 sorted_candidate_seeds=None,
                 seeds=None,
                 indicator=None,
                 parent_level=None,
                 scale=None):
        self.sorted_candidate_seeds = sorted_candidate_seeds
        self.seeds = seeds
        self.indicator = indicator
        self.parent_level = parent_level
        self.child_level = None
        self.scale = scale

    @property
    def is_used(self):
        return self.child_level is not None
    
    @property
    def name(self):
        pos = self.parent_level.buds.index(self)
        return f"{self.parent_level.name}-{pos}"
    

class BudCreator:

    def __init__(self,
                 seed_method,
                 level_probability,
                 scaler,
                 verbose):
        self.seed_method = getattr(self, seed_method)
        self.level_probability = level_probability
        self.scaler = scaler
        self.verbose = verbose

    def create(self,level,partition_indicator):
        sorted_candidate_seeds = self.sorted_candidate_seeds(level,
                                                             partition_indicator)
        if len(sorted_candidate_seeds) == 0:
            return None
        
        seeds = self.seed_method(level, sorted_candidate_seeds)
        threshold = seeds[0].performance
        indicator = level.indicator.child_indicator(threshold,
                                                    partition_indicator)
        scale = self.scaler.compute_scale(seeds)
        bud = Bud(sorted_candidate_seeds=sorted_candidate_seeds,
                  seeds=seeds,
                  indicator=indicator,
                  parent_level=level,
                  scale=scale
                  )
        return bud
    
    def create_trivial(self,level):
        level.trivial_bud = self.create(level, lambda x: 1)

    def create_inidicator_buds(self,level,partition_indicators):
        for partition_indicator in partition_indicators:
            bud = self.create(level, partition_indicator)
            if bud is None:
                self.print_empty_bud()
                level.empty_indicators.append(partition_indicator)
            else:
                level._buds.append(bud)
        self.print_partition(level)

    @verbose
    def print_partition(self, level):
        if len(level.buds) > 1:
            print(f"Level {level.name} partitioned into {len(level.buds)}.")

    @verbose
    def print_empty_bud(self):
        print("Empty bud.")

    @staticmethod
    def sorted_candidate_seeds(level,partition_indicator):
        samples = [sample
                   for sample in level.sample_list
                   if bool(partition_indicator(sample.array))]
        return sorted(samples, key=lambda x: x.performance)
    
    def seeds_prob_inclusive(self,level,sorted_candidate_seeds):
        seed_number = int(self.level_probability * level.level_size)
        thresh = sorted_candidate_seeds[-seed_number].performance
        return sorted([seed for seed in sorted_candidate_seeds if seed.performance >= thresh],
                      key=lambda x: x.performance)

    def seeds_prob_exclusive(self,level,sorted_candidate_seeds):
        seed_number = int(self.level_probability * level.level_size)
        return sorted_candidate_seeds[-seed_number:]

