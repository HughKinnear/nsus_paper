from examples.numerical_examples.two_dof.two_dof import two_dof, degenerate_two_dof
from examples.numerical_examples.experiment import multiple_experiments
import pickle

with open('examples/numerical_examples/two_dof/two_dof_ref_prob.pkl', 'rb') as file:
    two_dof_ref_prob = pickle.load(file)

exp_params = {'algo': 'nss',
              'ref_prob': two_dof_ref_prob,
              'detect_degenerate': degenerate_two_dof,
              'performance_function_name': 'two_dof',
              'seed_iters': 100,
              'is_bulk': False}

fixed_params = {'threshold': 0,
                'performance_function':two_dof,
                'dimension': 2,
                'scale': 1}

level_size_options = [750]
graph_size_options = [30]
max_branch_options = [2]


multiple_experiments('two_dof_nss_exp.pkl',
                      exp_params,
                      fixed_params,
                      level_size_options,
                      graph_size_options,
                      max_branch_options)