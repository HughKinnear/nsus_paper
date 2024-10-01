import networkx as nx
import numpy as np
from .partition import Partitioner
from networkx.algorithms.community import asyn_lpa_communities
from sklearn.metrics import balanced_accuracy_score
from collections import Counter


class HillValleyGraph:

    def __init__(self):
        self.graph = None
        self.eval_count = 0

    def cache_fit(self, X, y, cache, perf_function):
        if self.trivial_graph(X):
            return 
        i,j = np.tril_indices(len(X), k=-1)
        within_range, _ = self.history_and_within_range(cache,X,i,j)
        no_between_index = np.where(~np.any(within_range,axis=1))[0]
        for ind in no_between_index:
            perf_function((X[i][ind] + X[j][ind])/2)
            self.eval_count += 1
        within_range, hist_perfs = self.history_and_within_range(cache,X,i,j)
        perf_bools = (hist_perfs < np.minimum(y[i],y[j]).reshape(-1,1))
        adj_info = (~np.any(within_range & perf_bools,axis=1)).astype(int)
        self.adj_info_to_graph(adj_info,i,j,X)

    def fit(self, X, y, perf_function):
        if self.trivial_graph(X):
            return 
        i,j = np.tril_indices(len(X), k=-1)
        performances = np.array([perf_function((x_i+ x_j)/2) for x_i,x_j in zip(X[i],X[j])])
        adj_info = (performances >= np.minimum(y[i],y[j])).astype(int)
        self.adj_info_to_graph(adj_info,i,j,X)

    def trivial_graph(self, X):
        cond = len(X) == 1
        if cond:
            graph = nx.Graph()
            graph.add_nodes_from(range(len(X)))
            self.graph = graph
        return cond
    
    def adj_info_to_graph(self,adj_info,i,j,X):
        adj_matrix = np.zeros((len(X),len(X)))
        adj_matrix[i,j] = adj_info
        adj_matrix[j,i] = adj_info
        self.graph = nx.from_numpy_array(adj_matrix)

    @staticmethod
    def create_history_data(cache,X):
        hist_arrays = np.array([tup for tup in cache.keys()])
        hist_perfs = np.array([val for val in cache.values()])
        mask = ~((hist_arrays[:,None] == X).all(-1).any(1))
        return hist_arrays[mask], hist_perfs[mask]

    @staticmethod
    def create_within_range(end_points_i,end_points_j, mask_hist_arrays):
        min_points = np.minimum(end_points_i, end_points_j)
        max_points = np.maximum(end_points_i, end_points_j)
        min_points = min_points[:, np.newaxis, :]
        max_points = max_points[:, np.newaxis, :]
        hist_expanded = mask_hist_arrays[np.newaxis, :, :]
        return np.all((hist_expanded >= min_points) & (hist_expanded <= max_points), axis=2)

    def history_and_within_range(self,cache,X,i,j):
        hist_arrays, hist_perfs = self.create_history_data(cache,X)
        within_range = self.create_within_range(X[i],X[j],hist_arrays)
        return within_range, hist_perfs


class HillValleyGraphPartitioner(Partitioner):

    def __init__(self,
                 classifier_creator,
                 random_state,
                 max_branches,
                 graph_size,
                 iterations,
                 is_cache,
                 *args,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.classifier_creator = classifier_creator
        self.max_branches = max_branches
        self.graph_size = graph_size
        self.is_cache = is_cache
        self.random_state = random_state
        self.eval_count = 0
        self.iterations = iterations

    
    def partition(self,level):

        if len(level.non_used_buds) >= self.max_branches:
            return [self.trivial_indicator]
        
        adjusted_max_partition = self.max_branches - len(level.non_used_buds) + 1

        level_to_hvg(level,self.graph_size,self.random_state,self.is_cache)
        hvg = level.extra_info['hvg']
        self.eval_count += hvg.eval_count
        level.extra_info['graph'] = hvg.graph
            

        partitions = [list(asyn_lpa_communities(hvg.graph,seed=self.random_state))
                      for _ in range(self.iterations)]
        part_len_counts = Counter([len(part) for part in partitions])
        if part_len_counts[1] / self.iterations > 0.5:
            return [self.trivial_indicator]
        partitions = [part for part in partitions if  len(part) > 1]
        partitions = [shrink_partition(part,adjusted_max_partition,self.random_state)
                      for part in partitions]
            
        classifier = self.classifier_creator.create(level)
        scores = []
        for partition in partitions:
            X,y,_ = self.fit_classifier(level,partition,classifier)
            scores.append(balanced_accuracy_score(y,classifier.predict(X)))
        best_partition = partitions[np.argmax(scores)]

        X,y,sample_partition = self.fit_classifier(level,best_partition,classifier)
        level.extra_info['partition'] = best_partition
        level.extra_info['classifier'] = classifier
        level.extra_info['X'] = X
        level.extra_info['y'] = y
        level.extra_info['sample_partition'] = sample_partition
        return [self.indicator_factory(classifier.predict_single,label)
                for label in range(len(best_partition))]
    
    @staticmethod
    def indicator_factory(classifier, label):
        def indicator(x):
            return int(classifier(x) == label)
        return indicator
    
    @staticmethod
    def trivial_indicator(x):
        return 1

    def fit_classifier(self, level, partition, classifier):
        sample_partition = [[level.extra_info['graph_samples'][ind] for ind in part]
                             for part in partition]
        X = np.array([samp.array for part in sample_partition for samp in part])
        y = np.array([i for i, part_set in enumerate(sample_partition) for _ in range(len(part_set))])
        classifier.fit(X,y)
        return X,y,sample_partition
        
    
def level_to_hvg(level,graph_size,random_state,is_cache):
    choose_samples(level,graph_size,random_state)
    graph_samples = level.extra_info['graph_samples']
    perf_function = level.indicator.performance_function
    arrays = np.array([samp.array for samp in graph_samples])
    perfs = np.array([samp.performance for samp in graph_samples])
    hvg = HillValleyGraph()
    if is_cache:
        cache = perf_function.cache
        hvg.cache_fit(arrays,perfs,cache,perf_function)
    else:
        hvg.fit(arrays,perfs,perf_function)
    level.extra_info['hvg'] = hvg


def choose_samples(level,graph_size,random_state):
    seed_array = np.array([chain[0].array for chain in level.chain_list])
    unique_seeds = np.unique(seed_array, axis=0)
    chain_group_indicies = [[] for _ in range(len(unique_seeds))]
    for i in range(len(seed_array)):
        chain_group_indicies[np.where(np.all(unique_seeds == seed_array[i],axis=1))[0][0]].append(i)
    chain_groups_non_unique = [sum([level.chain_list[ind] for ind in inds],[])
                            for inds in chain_group_indicies]
    chain_groups = []
    for chain_group in chain_groups_non_unique:
        unique_inds = np.unique([samp.array for samp in chain_group], axis=0, return_index=True)[1]
        chain_groups.append([chain_group[ind] for ind in unique_inds])
    for chain_group in chain_groups:
        random_state.shuffle(chain_group)
    random_state.shuffle(chain_groups)
    graph_sample_list = []
    while any(chain_group for chain_group in chain_groups):
        for chain_group in chain_groups:
            if chain_group:
                graph_sample_list.append(chain_group.pop())
    graph_size = graph_size // len(level.non_used_buds)
    graph_samples = graph_sample_list[:graph_size]
    level.extra_info['graph_samples'] = graph_samples


def shrink_partition(partition, n,rng):
    partition = list(partition)
    while len(partition) > n:
        idxa, idxb = rng.choice(range(len(partition)),size=2,replace=False)
        partition[idxa] = partition[idxa].union(partition[idxb])
        partition.pop(idxb)
    return partition
