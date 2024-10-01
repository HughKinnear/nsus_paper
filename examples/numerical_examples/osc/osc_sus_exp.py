from examples.numerical_examples.osc.osc import osc, degenerate_osc
from examples.numerical_examples.experiment import multiple_experiments
import pickle

with open('examples/numerical_examples/osc/osc_ref_prob.pkl', 'rb') as file:
    osc_ref_prob = pickle.load(file)


exp_params = {'algo': 'sus',
              'ref_prob': osc_ref_prob,
              'detect_degenerate': degenerate_osc,
              'performance_function_name': 'osc',
              'seed_iters': 100,
              'is_bulk': False}

fixed_params = {'threshold': 0,
                'performance_function':osc(),
                'dimension': 101,
                'scale': 1}

level_size_options = [800]


multiple_experiments('osc_sus_exp.pkl',
                      exp_params,
                      fixed_params,
                      level_size_options,)