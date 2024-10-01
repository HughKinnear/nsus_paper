import numpy as np
from itertools import product
from scipy.stats import norm


def failure_to_reliability(failure):
    return -norm.ppf(failure)

def reliability_to_failure(reliability):
    return norm.cdf(-reliability)

def probability_of_indicator(indicator, level):
    return sum(indicator(level.sample_list)) / level.level_size

def exceedance_probability(bss, threshold):
    return sum([estimate_trimmed_leaf_probability(threshold, leaf)
                for leaf in trimmed_leaves(bss, threshold)])

def estimate_leaf_probability(thresh, leaf):
    trimmed_leaf = trim_leaf(leaf, thresh)
    return estimate_trimmed_leaf_probability(thresh, trimmed_leaf)

def trim_leaf(leaf, threshold):
    branch = leaf.branch
    i = -1
    while threshold < branch[i].indicator.threshold:
        i -= 1
    return branch[i]

def estimate_trimmed_leaf_probability(threshold, leaf):
    branch = leaf.branch
    return (np.prod([probability_of_indicator(level.indicator, level.parent)
                     for level in branch[1:]])
            * probability_of_indicator(leaf.indicator.new_threshold(threshold),
                                       leaf))

def trimmed_leaves(nss, threshold):
    leaves = []
    potential_leaves = [nss.initial_level]
    while potential_leaves:
        leaf = potential_leaves.pop()
        if not leaf.children:
            leaves.append(leaf)
            continue
        cond = any([child.indicator.threshold > threshold
                    for child in leaf.children])
        if cond:
            leaves.append(leaf)
        else:
            potential_leaves.extend(leaf.children)
    return leaves


def cov(bss, threshold):
    leaves = trimmed_leaves(bss, threshold)
    leave_it = range(len(leaves))
    prod_ij = list(product(leave_it, repeat=2))

    p = [estimate_trimmed_leaf_probability(threshold,
                                           leaf)
         for leaf in leaves]
    w_total = sum([p[i] * p[j]
                   for i, j in prod_ij])
    w_dict = {i: {j: (p[i] * p[j])/w_total for j in leave_it}
              for i in leave_it}

    common_ancestors = {i: {j: find_common_ancestor(leaves[i], leaves[j])
                            for j in leave_it}
                        for i in leave_it}

    big_sum = []
    for i, j in prod_ij:
        if i == j:
            temp_thresh = threshold
        else:
            temp_thresh = common_ancestors[i][j].indicator.threshold
        big_sum.append(w_dict[i][j]
                       * (leaf_cov(common_ancestors[i][j], temp_thresh)**2))
    return np.sqrt(sum(big_sum))

def find_common_ancestor(leaf_a, leaf_b):
    return [node
            for node in leaf_a.branch
            if node in leaf_b.branch][-1]

def leaf_cov(leaf, threshold):
    cov_square = [level_cov(level.parent, level.indicator) ** 2
                  for level in leaf.branch[1:]]
    last_indicator = leaf.indicator.new_threshold(threshold)
    cov_square.append(level_cov(leaf, last_indicator) ** 2)
    return np.sqrt(sum(cov_square))

def level_cov(level, indicator):
    prob = probability_of_indicator(indicator, level)
    if prob == 1:
        return 0
    if level.order == 1:
        return np.sqrt((1 - prob) / (level.level_size * prob))
    else:
        gamma = level_gamma(level, indicator)
        return np.sqrt(((1 - prob) / (level.level_size * prob)) * (1 + gamma))

def level_gamma(level, indicator):
    n_c = len(level.chain_list)
    n = level.level_size
    n_s = int(n // n_c)

    prob = probability_of_indicator(indicator, level)
    f_indicator_list = indicator(level.sample_list)
    indicator_list = [f_indicator_list[i * n_s:(i + 1) * n_s]
                      for i in range(n_c)]

    r_i_list = []
    for i in range(n_s):
        sums = sum([sum([chain[i_dash] * chain[i_dash + i]
                         for i_dash in range(n_s - i)])
                    for chain in indicator_list])
        r_i_list.append((sums / (n - (i * n_c))) - (prob ** 2))

    
    if r_i_list[0] > 0:
        gamma = 2 * (sum([(1 - (i / n_s)) * (r_i_list[i+1] / r_i_list[0])
                          for i in range(n_s-1)]))
    else:
        gamma = np.inf
    return gamma

def level_ess(level, indicator):
    if level.order == 1:
        return level.level_size
    else:
        gamma = level_gamma(level, indicator)
        return level.level_size / (1 + gamma)  

def efficiency(level,indicator):
    return (1+level_gamma(level,indicator))**-1

def leaf_efficiency(leaf):
     branch = leaf.branch[1:]
     buds = [level.initial_bud for level in branch[1:]] + [leaf.trivial_bud]
     return [efficiency(level,bud.indicator) for level,bud in zip(branch,buds)]

