from nsus.estimate import failure_to_reliability
import numpy as np
import pickle
import nsus.estimate as est
from nsus.implementation import SubsetSimulation, HillValleyGraphNSuS
from tqdm import tqdm
import pandas as pd
from itertools import product
import warnings


def experiment(algo,
               ref_prob,
               detect_degenerate,
               performance_function_name,
               seed_iters,
               params,
               is_bulk):
    ref_reliability = failure_to_reliability(ref_prob)
    failure_probs = []
    performance_evals = []
    degenerate_bools = []
    hv_evals = []
    if is_bulk:
        bulk_failure_probs = []
        bulk_performance_evals = []
        bulk_degenerate_bools = []
    params['level_probability'] = 0.1
    params['verbose'] = False
    for seed in tqdm(range(seed_iters)):
        ss = (SubsetSimulation(seed=seed,**params) if algo == 'sus'
              else HillValleyGraphNSuS(seed=seed,**params))
        ss.run()
        failure_probs.append(est.exceedance_probability(ss,params['threshold']))
        performance_evals.append(ss.performance_function.eval_count)
        degenerate_bools.append(detect_degenerate(ss))
        hv_eval = ss.partitioner.eval_count if algo == 'nss' else 0
        hv_evals.append(hv_eval)
        if is_bulk:
            ss.bulk(params['threshold'],0.01)
            bulk_failure_probs.append(est.exceedance_probability(ss,params['threshold']))
            bulk_performance_evals.append(ss.performance_function.eval_count)
            bulk_degenerate_bools.append(detect_degenerate(ss))
    
    
    reliabilities = [failure_to_reliability(failure_prob)
                     for failure_prob in failure_probs]
    del params['performance_function']
    
    result_dict = {
        'ref_prob': ref_prob,
        'ref_reliability': ref_reliability,
        'joint_rels': np.array(reliabilities),
        'joint_probs': np.array(failure_probs),
        'joint_evals': np.array(performance_evals),
        'degenerate_bools': np.array(degenerate_bools),
        'hv_evals': np.array(hv_evals),
        'algo': algo,
        'performance_function_name': performance_function_name,
        'iterations': seed_iters,
        **params
    }
    if is_bulk:
        bulk_reliabilities = [failure_to_reliability(failure_prob)
                              for failure_prob in bulk_failure_probs]
        bulk_result_dict = {
            'ref_prob': ref_prob,
            'ref_reliability': ref_reliability,
            'joint_rels': np.array(bulk_reliabilities),
            'joint_probs': np.array(bulk_failure_probs),
            'joint_evals': np.array(bulk_performance_evals),
            'degenerate_bools': np.array(bulk_degenerate_bools),
            'hv_evals': np.array(hv_evals),
            'algo': algo + '_bulk',
            'performance_function_name': performance_function_name,
            'iterations': seed_iters,
            **params}

        return (result_dict, bulk_result_dict)
    else:
        return result_dict


def multiple_experiments(filename,
                         exp_params,
                         fixed_params,
                         level_size_options,
                         graph_size_options=[None],
                         max_branches_options=[None]):
    iterator = product(level_size_options,
                        graph_size_options,
                        max_branches_options)
    results = []
    for dyn_param in iterator:
        params = {**fixed_params,
                  'level_size': dyn_param[0]}
        if not dyn_param[1] is None:
            params['graph_size'] = dyn_param[1]
            params['max_branches'] = dyn_param[2]
        results.append(experiment(**exp_params, params=params))
    with open(filename, 'wb') as file:
        pickle.dump(results, file)


class ExperimentResult:

    def __init__(self,result_dict):
        for key, value in result_dict.items():
            setattr(self, key, value)
        self.groups = ['joint','degen','non_degen']
        self.quantities = ['prob','rel','eval']
        self.stats = ['mean','cov']
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for c in product(self.groups[1:],self.quantities):
                bool_array = (self.degenerate_bools if c[0] == 'degen'
                            else ~self.degenerate_bools)
                setattr(self, f'{c[0]}_{c[1]}s',
                        getattr(self,f'joint_{c[1]}s')[bool_array])
            for c in product(self.groups,self.quantities[:-1],self.stats):
                array = getattr(self,f'{c[0]}_{c[1]}s')
                setattr(self, f'{c[0]}_{c[1]}_{c[2]}', np.mean(array) if c[2] == 'mean'
                        else np.std(array)/np.mean(array))   
            for g in self.groups:
                setattr(self, f'{g}_eval_mean', np.mean(getattr(self,f'{g}_evals')))
        
            if self.algo == 'nss':
                self.joint_hv_evals = self.hv_evals
                self.degen_hv_evals = self.hv_evals[self.degenerate_bools]
                self.non_degen_hv_evals = self.hv_evals[~self.degenerate_bools]
                self.joint_hv_eval_mean = np.mean(self.joint_hv_evals)
                self.degen_hv_eval_mean = np.mean(self.degen_hv_evals)
                self.non_degen_hv_eval_mean = np.mean(self.non_degen_hv_evals)
                self.joint_non_hv_eval_mean = self.joint_eval_mean - self.joint_hv_eval_mean
                self.degen_non_hv_eval_mean = self.degen_eval_mean - self.degen_hv_eval_mean
                self.non_degen_non_hv_eval_mean = self.non_degen_eval_mean - self.non_degen_hv_eval_mean
            else:
                self.joint_non_hv_eval_mean = self.joint_eval_mean
                self.degen_non_hv_eval_mean = self.degen_eval_mean
                self.non_degen_non_hv_eval_mean = self.non_degen_eval_mean
                self.joint_hv_eval_mean = 0
                self.degen_hv_eval_mean = 0
                self.non_degen_hv_eval_mean = 0

        self.degen_percentage = np.mean(self.degenerate_bools)
        self.non_degen_percentage = 1 - self.degen_percentage
    
    def table(self):
        df = pd.DataFrame({
            'Weight': [1,self.degen_percentage,self.non_degen_percentage],
            'Failure Mean': [f'{getattr(self,f'{g}_prob_mean'):.2e}' for g in self.groups],
            'Failure CoV': [f'{getattr(self,f'{g}_prob_cov'):.2f}' for g in self.groups],
            'Performance Eval Mean': [f'{getattr(self,f'{g}_non_hv_eval_mean'):.0f} + {getattr(self,f'{g}_hv_eval_mean'):.0f}'
                                      for g in self.groups]
        })
        df.index = ['Joint', 'Degenerate', 'Non-Degenerate']
        return df
    
    def info_print(self):
        print(f'Algorithm: {self.algo}')
        print(f'Performance Function: {self.performance_function_name}')
        print(f'Level Size: {self.level_size}')
        if self.algo == 'nss':
            print(f'Graph Size: {self.graph_size}')
            print(f'Max Branches: {self.max_branches}')
        print(f'Scale: {self.scale}')
        print(f'Dimension: {self.dimension}')
        print(f'Reference Probability: {self.ref_prob:.2e}')
        print(f'Reference Reliability: {self.ref_reliability:.2f}')

    def print_table(self):
        self.info_print()
        print(self.table())



def load_results(filename):
    with open(filename, "rb") as file:
        results = pickle.load(file)
    return [ExperimentResult(result) 
            if not type(result) == tuple
            else (ExperimentResult(result[0]), ExperimentResult(result[1]))
            for result in results]
