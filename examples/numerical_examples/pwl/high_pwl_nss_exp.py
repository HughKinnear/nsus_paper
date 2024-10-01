from examples.numerical_examples.pwl.pwl import high_pwl, degenerate_pwl
from examples.numerical_examples.experiment import multiple_experiments
import pickle

with open('examples/numerical_examples/pwl/pwl_ref_prob.pkl', 'rb') as file:
    pwl_ref_prob = pickle.load(file)

exp_params = {'algo': 'nss',
              'ref_prob': pwl_ref_prob,
              'detect_degenerate': degenerate_pwl,
              'performance_function_name': 'high_pwl',
              'seed_iters': 100,
              'is_bulk': False}

fixed_params = {'threshold': 0,
                'performance_function':high_pwl,
                'dimension': 100,
                'scale': 1}

level_size_options = [500,1000,1500,2000,2500,3000]
graph_size_options = [15,20,25]
max_branches_options = [2]


multiple_experiments('high_pwl_nss_exp.pkl',
                      exp_params,
                      fixed_params,
                      level_size_options,
                      graph_size_options,
                      max_branches_options)