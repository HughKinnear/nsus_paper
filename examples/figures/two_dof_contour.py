import matplotlib.pyplot as plt
from examples.figures.utils import contour_plot, plot_level, colors
from examples.numerical_examples.two_dof.two_dof import two_dof
from nsus.implementation import HillValleyGraphNSuS, SubsetSimulation
from matplotlib.lines import Line2D


##########

fig_legend = plt.figure(figsize=(1, 1))
plt.axis('off')
legend_labels = ['Partition boundary', 'Limit state surface', 'Contours','Level samples']
proxy_artists = ([Line2D([0], [0], color=color, lw=2) for color in ['black', colors[3]]]
                    + [Line2D([0], [0], color=colors[3], lw=2,linestyle='--')]
                    + [plt.scatter([], [], color=colors[0], s=15)])
legend = fig_legend.legend(proxy_artists,legend_labels,ncols=2)
fig_legend.canvas.draw()
fig_legend.savefig('examples/figures/images/contour_legend.pdf',bbox_inches='tight')
plt.show()

############


sus = SubsetSimulation(performance_function=two_dof,
                           dimension=2,
                           level_size=750,
                           threshold=0,
                           level_probability=0.1,
                           seed=5,
                           scale=1,
                           verbose=False)

sus.run()

nss = HillValleyGraphNSuS(performance_function=two_dof,
                           dimension=2,
                           level_size=750,
                           threshold=0,
                           level_probability=0.1,
                           seed=3,
                           graph_size=30,
                           max_branches=2,
                           scale=1,
                           verbose=False)

nss.run()


##########

plt.style.use('ggplot')


plt.figure(figsize=(7, 7))
plt.grid(False)
contour_plot((-6,6), (-6,6), 0.1, two_dof, color=colors[3],levels=[-0.007,-0.005,-0.004,-0.003,0])
plot_level(6,sus,colors[0])
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.tick_params(labelsize=15)
plt.savefig('examples/figures/images/two_dof_contour_sus.pdf',bbox_inches='tight')
plt.show()


############



plt.figure(figsize=(7, 7))
plt.grid(False)

contour_plot((-6,6), (-6,6), 0.1, two_dof, color=colors[3],levels=[-0.007,-0.005,-0.004,-0.003,0])
plot_level(11,nss,colors[0])
level_1 = nss.find('1')
classifier = lambda x: level_1.extra_info['classifier'].predict_single(x)
contour_plot((-6,6), (-6,6), 0.01, classifier,color='black', levels=None)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.tick_params(labelsize=15)
plt.savefig('examples/figures/images/two_dof_contour_nss.pdf',bbox_inches='tight')
plt.show()



