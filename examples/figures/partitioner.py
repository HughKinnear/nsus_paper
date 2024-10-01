import matplotlib.pyplot as plt
import numpy as np
from nsus.implementation import HillValleyGraphNSuS
from examples.figures.utils import contour_plot
from examples.numerical_examples.pwl.pwl import pwl
import networkx as nx

nss = HillValleyGraphNSuS(performance_function=pwl,
                       level_probability=0.1,
                       level_size=500,
                       dimension=2,
                       seed=6,
                       threshold=0,
                       scale=1,
                       graph_size=20,
                       max_branches=2,
                       verbose=False,
                       is_cache=False)


nss.run()

level = nss.find('2')
info = level.extra_info
graph_samples = info['graph_samples']
graph = info['graph']
adj = nx.adjacency_matrix(graph)
sample_partition = info['sample_partition']
plt.style.use('ggplot')

colors = [
    '#e66101',
    '#fdb863',
    '#b2abd2',
    '#5e3c99'
]

######

plt.figure(figsize=(10, 7))

level_array = np.array([samp.array for samp in level.sample_list])
plotter = level_array.T
plt.scatter(plotter[0],plotter[1],
            s=10,label = 'Level samples', color=colors[2], edgecolor='black')

plotter = np.array([samp.array for samp in graph_samples]).T
plt.scatter(plotter[0],plotter[1],s=250, color=colors[1], label = 'Graph samples',edgecolor='black')

plt.xlabel(u'x\u2081',fontsize=20)
plt.ylabel(u'x\u2082',fontsize=20)

plt.legend(fontsize=18,frameon=True,facecolor='white',loc='lower left')
plt.xlim(-2,3.2)
plt.ylim(-4,4)
plt.tick_params(labelsize=15)


plt.savefig('examples/figures/images/resampled_points.pdf',bbox_inches='tight')
plt.show()


#####

plt.figure(figsize=(10, 7))

plotter = np.array([samp.array for samp in graph_samples]).T

is_label = True
for i in range(len(plotter.T)):
    for j in range(i + 1, len(plotter.T)):
        if adj[i, j] == 1:
            label = 'Graph edges' if is_label else ''
            plt.plot([plotter.T[i, 0],plotter.T[j, 0]], [plotter.T[i, 1], plotter.T[j, 1]], 
                        label=label,
                     color='black',zorder=1)
            is_label = False
            
plt.scatter(plotter[0],plotter[1],s=250, color=colors[1], label = 'Graph samples',edgecolor='white')

plt.xlabel(u'x\u2081',fontsize=20)
plt.ylabel(u'x\u2082',fontsize=20)
plt.tick_params(labelsize=15)

plt.xlim(-2,3.2)
plt.ylim(-4,4)


plt.legend(fontsize=18,frameon=True,facecolor='white',loc='lower left')


plt.savefig('examples/figures/images/hill_valley_graph.pdf',bbox_inches='tight')
plt.show()

# #####

plt.figure(figsize=(10, 7))

plotter = np.array([samp.array for samp in graph_samples]).T

is_label = True
for i in range(len(plotter.T)):
    for j in range(i + 1, len(plotter.T)):
        if adj[i, j] == 1:
            label = 'Graph edges' if is_label else ''
            plt.plot([plotter.T[i, 0],plotter.T[j, 0]], [plotter.T[i, 1], plotter.T[j, 1]],
                     label=label,
                     color='black',
                     zorder=1)
            is_label = False
                       
for part_set,c,label in zip(sample_partition,[colors[0],colors[3]],['Niche 0','Niche 1']):
    plotter = np.array([samp.array for samp in part_set]).T
    plt.scatter(plotter[0],plotter[1],s=250,zorder=2,color=c,label=label,edgecolor='white')
    
plt.xlabel(u'x\u2081',fontsize=20)
plt.ylabel(u'x\u2082',fontsize=20)
plt.tick_params(labelsize=15)



plt.legend(fontsize=18,frameon=True,facecolor='white',loc='lower left')

plt.xlim(-2,3.2)
plt.ylim(-4,4)

plt.savefig('examples/figures/images/community_detection.pdf',bbox_inches='tight')
plt.show()

# #####

plt.figure(figsize=(10, 7))

plt.plot(0,0,label='Partition boundary',color='black')
classifier = lambda x: info['classifier'].predict_single(x)
for part_set,c,label in zip(sample_partition,[colors[0],colors[3]],['Niche 0','Niche 1']):
    plotter = np.array([samp.array for samp in part_set]).T
    plt.scatter(plotter[0],plotter[1],s=250,zorder=2,color=c,label=label,edgecolor='white')
    
contour_plot((-2,3.2), (-4,4), 0.01, classifier, levels=None)


plt.legend(fontsize=18,frameon=True,facecolor='white',loc='lower left')

plt.xlabel(u'x\u2081',fontsize=20)
plt.ylabel(u'x\u2082',fontsize=20)
plt.tick_params(labelsize=15)

plt.xlim(-2,3.2)
plt.ylim(-4,4)

plt.savefig('examples/figures/images/classifier.pdf',bbox_inches='tight')
plt.show()

# #####

# plt.figure(figsize=(10, 7))


# classifier = lambda x: info['classifier'].predict_single(x)

# contour_plot((-2,3.2), (-4,4), 0.01, classifier, levels=None)
    
# level_array = np.array([samp.array for samp in level.sample_list])
# plotter = level_array.T
# plt.scatter(plotter[0],plotter[1],
#             s=10,label = 'Level samples', color=colors[2], edgecolor='black')

# plt.xlabel(u'x\u2081',fontsize=15)
# plt.ylabel(u'x\u2082',fontsize=15)

# plt.xlim(-2,3.2)
# plt.ylim(-4,4)

# plt.savefig('examples/figures/images/classifier_original.pdf',bbox_inches='tight')
# plt.show()



