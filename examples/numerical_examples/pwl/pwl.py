import numpy as np
from examples.numerical_examples.utils import convert_array, covnert_performance_function

def pwl(x):
    if x[0] > 3.5:
        g_1 = 4 - x[0]
    else:
        g_1 = 0.85 - (0.1 * x[0])
    if x[1] > 2:
        g_2 = 0.5 - (0.1 * x[1])
    else:
        g_2 = 2.3 - x[1]
    return -min([g_1, g_2])

high_pwl = covnert_performance_function(pwl,2,50)

def degenerate_pwl(ss):
    arrays = np.array([convert_array(sample.array) for sample in ss.all_samples])
    return not ((arrays[:,0] >= 4) & (arrays[:,1] <= 2)).any()