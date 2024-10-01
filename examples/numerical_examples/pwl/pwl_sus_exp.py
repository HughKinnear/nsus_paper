from examples.numerical_examples.pwl.pwl import pwl, degenerate_pwl
from examples.numerical_examples.experiment import multiple_experiments
import pickle

with open('examples/numerical_examples/pwl/pwl_ref_prob.pkl', 'rb') as file:
    pwl_ref_prob = pickle.load(file)

exp_params = {'algo': 'sus',
              'ref_prob': pwl_ref_prob,
              'detect_degenerate': degenerate_pwl,
              'performance_function_name': 'pwl',
              'seed_iters': 100,
              'is_bulk': False}

fixed_params = {'threshold': 0,
                'performance_function':pwl,
                'dimension': 2,
                'scale': 1}

level_size_options = [500,1000,1500,2000,2500,3000]


multiple_experiments('pwl_sus_exp.pkl',
                      exp_params,
                      fixed_params,
                      level_size_options)