from examples.numerical_examples.mix.mix import mix, degenerate_mix
from examples.numerical_examples.experiment import multiple_experiments
import pickle

with open('examples/numerical_examples/mix/mix_ref_prob.pkl', 'rb') as file:
    mix_ref_prob = pickle.load(file)

exp_params = {'algo': 'sus',
              'ref_prob': mix_ref_prob,
              'detect_degenerate': degenerate_mix,
              'performance_function_name': 'mix',
              'seed_iters': 100,
              'is_bulk': False}

fixed_params = {'threshold': 0,
                'performance_function':mix,
                'dimension': 2,
                'scale': 1}

level_size_options = [750]

multiple_experiments('mix_sus_exp.pkl',
                      exp_params,
                      fixed_params,
                      level_size_options)