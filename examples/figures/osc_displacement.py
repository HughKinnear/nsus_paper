from examples.numerical_examples.osc.osc import osc
from nsus.implementation import HillValleyGraphNSuS
import matplotlib.pyplot as plt
from examples.figures.utils import colors

osc_function = osc()

nss = HillValleyGraphNSuS(performance_function=osc_function,
                           dimension=101,
                           level_size=800,
                           threshold=0,
                           level_probability=0.1,
                           seed=1,
                           graph_size=15,
                           max_branches=2,
                           scale=1,
                           verbose=True)

nss.run()

plt.style.use('ggplot')
y_cache = nss.performance_function.non_cache_performance_function.y_cache
plt.figure(figsize=(10, 7))


level = nss.find('2')
niche_1 = [samp.array for samp in level.buds[0].sorted_candidate_seeds]
for i,arr in enumerate(niche_1):
    cache_results = y_cache[tuple(arr)]
    x = cache_results['t']
    y = cache_results['y'][0]
    if i == 0:
        plt.plot(x,y,color=colors[0],alpha=1, label='Niche 0')
    else:
        plt.plot(x,y,color=colors[0],alpha=0.1)

niche_2 = [samp.array for samp in level.buds[1].sorted_candidate_seeds]
for i,arr in enumerate(niche_2):
    cache_results = y_cache[tuple(arr)]
    x = cache_results['t']
    y = cache_results['y'][0]
    if i == 0:
        plt.plot(x,y,color=colors[3],alpha=1, label='Niche 1')
    else:
        plt.plot(x,y,color=colors[3],alpha=0.1)

plt.axhline(1.2,color='black',label='Displacement threshold')
plt.axhline(-1.2,color='black')
plt.xlim(0,1)
plt.ylim(-1.4,1.4)
plt.grid(False)
plt.xlabel('Time',fontsize=20)
plt.ylabel('Displacement',fontsize=20)
plt.tick_params(labelsize=18)
# plt.legend(facecolor='white',bbox_to_anchor=(0.3, 0.9))
plt.savefig('examples/figures/images/osc_branch.pdf',bbox_inches='tight')
plt.show()







plt.figure(figsize=(10, 7))
ax = plt.gca()

level = nss.find('15')
niche_1 = [samp.array for samp in level.sample_list]
for i,arr in enumerate(niche_1):
    cache_results = y_cache[tuple(arr)]
    x = cache_results['t']
    y = cache_results['y'][0]
    if i == 0:
        plt.plot(x,y,color=colors[0],alpha=1, label='Niche 0')
    else:
        plt.plot(x,y,color=colors[0],alpha=0.1)

level = nss.find('16')
niche_2 = [samp.array for samp in level.sample_list]
for i,arr in enumerate(niche_2):
    cache_results = y_cache[tuple(arr)]
    x = cache_results['t']
    y = cache_results['y'][0]
    if i == 0:
        plt.plot(x,y,color=colors[3],alpha=1, label='Niche 1')
    else:
        plt.plot(x,y,color=colors[3],alpha=0.1)

plt.axhline(1.2,color='black',label='Displacement threshold')
plt.axhline(-1.2,color='black')
plt.xlim(0,1)
plt.ylim(-1.4,1.4)
plt.grid(False)
plt.xlabel('Time',fontsize=20)
plt.ylabel('Displacement',fontsize=20)
plt.tick_params(labelsize=18)
# plt.legend(facecolor='white',bbox_to_anchor=(0.3, 0.9))
plt.savefig('examples/figures/images/osc_fail.pdf',bbox_inches='tight')
plt.show()


plt.style.use('default')
plt.figure(figsize=(1,1))
plt.axis('off')
plt.legend(*ax.get_legend_handles_labels(),ncols=3)
plt.savefig('examples/figures/images/osc_legend.pdf',bbox_inches='tight')
plt.show()
