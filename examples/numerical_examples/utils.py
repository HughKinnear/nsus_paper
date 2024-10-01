import numpy as np

def covnert_performance_function(perf_function,orig_dim,new_dim_multiplier):

    def new_performance_function(x):
        new_input = [sum(x[new_dim_multiplier*i:new_dim_multiplier*(i+1)])
                     / np.sqrt(new_dim_multiplier)
                     for i in range(orig_dim)]
        return perf_function(np.array(new_input))
    
    return new_performance_function


def convert_array(x):
    half_dim = len(x) // 2
    x_1 = sum(x[:half_dim]) / np.sqrt(half_dim)
    x_2 = sum(x[half_dim:]) / np.sqrt(half_dim)
    return np.array([x_1,x_2])