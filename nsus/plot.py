import numpy as np
from anytree import PreOrderIter, RenderTree
from matplotlib import pyplot as plt
from . import estimate as est
import networkx as nx


def all_levels(bss):
    iter = sorted([level for level in PreOrderIter(bss.initial_level)],
                  key=lambda x: int(x.name))
    for level in iter:
        plotter = np.array([sample.array for sample in level.sample_list]).T
        plt.scatter(plotter[0], plotter[1])


def ccdf(bss, thresh, num_points):
    smallest_performance = bss.initial_level.sorted_list[0].performance
    perfs = np.linspace(smallest_performance, thresh, num_points)
    log_probs = np.array([np.log(est.exceedance_probability(bss, perf)) for perf in perfs])
    plt.plot(perfs, log_probs)
    plt.xlabel('Performance')
    plt.ylabel('Log Probability')
    plt.title('Complementary Cumulative Distribution Function')


def branch_ccdf(leaf, thresh, num_points):
    smallest_performance = leaf.branch[0].sorted_list[0].performance
    perfs = np.linspace(smallest_performance, thresh, num_points)
    log_probs = np.array([np.log(est.estimate_leaf_probability(perf, leaf)) for perf in perfs])
    plt.plot(perfs, log_probs)
    plt.xlabel('Performance')
    plt.ylabel('Log Probability')
    plt.title('Complementary Cumulative Distribution Function')


def render_tree(bss):
    print(RenderTree(bss.initial_level).by_attr())


def graph_partition_plot(g, partition):
    part_indicator = [0] * len(g.nodes)
    for i, partition_set in enumerate(partition):
        for node in partition_set:
            part_indicator[node] = i
    nx.draw(g,node_color=part_indicator)


def hvg_plot(nss,level_num):
    
    level = nss.find(level_num)
    info = level.extra_info
    g = info['graph']
    partition = info['partition']
    
    child_names = ['Bud ' + bud.child_level.name for bud in level.buds]
    color_map = {label: plt.cm.tab10(i) for i, label in enumerate(child_names)}
    labels = [0] * len(g.nodes)
    for i, partition_set in zip(child_names,partition):
        for node in partition_set:
            labels[node] = i
    node_colors = [color_map[labels[i]] for i in range(len(g.nodes))]
    nx.draw(g,node_color=node_colors)
    for label in child_names:
        plt.scatter([], [], c=[color_map[label]], label=label)
    plt.legend()

    for bud in level.buds:
        print(f'Bud {bud.child_level.name}:')
        print(f'  seeds: {len(bud.seeds)}')
        print(f'  candidates: {len(bud.sorted_candidate_seeds)}')
        performances = [samp.performance for samp in bud.sorted_candidate_seeds]
        mean = np.mean(performances)
        min_perf = np.min(performances)
        max_perf = np.max(performances)
        print(f'  mean performance: {mean}')
        print(f'  min performance: {min_perf}')
        print(f'  max performance: {max_perf}') 

