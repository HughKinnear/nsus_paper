import numpy as np

def mix(x):
    means = np.array([[3,3],
                  [2,-2],
                  [-2,2],
                  [-2,-2]])
    weights = [0.4, 0.2, 0.2, 0.2]
    diffs = x - means
    exps = -0.5 * np.einsum('ij,ij->i',diffs,diffs)
    vals = (1 / (2 * np.pi)) * np.exp(exps)
    return np.dot(weights,vals) - 0.04


def degenerate_mix(ss):
    arrays = np.array([sample.array for sample in ss.all_samples])
    pos_bool = (arrays[:,0] >= 0) & (arrays[:,1] >= 0)
    perf_array = np.array([sample.performance for sample in ss.all_samples])
    perf_bool = perf_array > 0
    return not (pos_bool & perf_bool).any()