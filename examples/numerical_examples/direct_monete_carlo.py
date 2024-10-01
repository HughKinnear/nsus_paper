import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm


class DirectMonteCarlo:

    def __init__(self,
                 dim,
                 performance_function,
                 sample_size,
                 chunks,
                 is_sample_save,
                 seed):
        self.dim = dim
        self.performance_function = performance_function
        self.sample_size = sample_size
        self.seed = seed
        self.random_state = np.random.default_rng(seed)
        self.chunks = chunks
        self.chunk_size = sample_size // chunks
        self.performances = []
        self.is_sample_save = is_sample_save
        if self.is_sample_save:
            self.samples = []
    
    def compute(self):
        for _ in tqdm(range(self.chunks)):
            samples = multivariate_normal.rvs(mean=np.zeros(self.dim),
                                               random_state=self.random_state,
                                               size=self.chunk_size)
            for sample in samples:
                self.performances.append(self.performance_function(sample))
                if self.is_sample_save:
                    self.samples.append(sample)

    def threshold(self,threshold):
        return sum(np.array(self.performances) > threshold) / self.sample_size