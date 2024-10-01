import matplotlib.pyplot as plt
from examples.figures.utils import contour_plot, plot_level, colors
from examples.numerical_examples.mix.mix import mix
from nsus.implementation import HillValleyGraphNSuS,SubsetSimulation



sus = SubsetSimulation(performance_function=mix,
                           dimension=2,
                           level_size=750,
                           threshold=0,
                           level_probability=0.1,
                           seed=5,
                           scale=1,
                           verbose=False)

sus.run()


nss = HillValleyGraphNSuS(performance_function=mix,
                           dimension=2,
                           level_size=750,
                           threshold=0,
                           level_probability=0.1,
                           seed=2,
                           graph_size=25,
                           max_branches=4,
                           scale=1,
                           verbose=False)

nss.run()

plt.style.use('ggplot')


#########

plt.figure(figsize=(7, 7))
plt.grid(False)
contour_plot((-4.2,4.2), (-4.2,4.2), 0.1, mix, color=colors[3],levels=[-0.025,-0.017,-0.01,0])
plot_level(6,sus,colors[0])
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.tick_params(labelsize=15)
plt.savefig('examples/figures/images/mix_contour_sus.pdf',bbox_inches='tight')
plt.show()


############


plt.figure(figsize=(7, 7))
plt.grid(False)
contour_plot((-4.2,4.2), (-4.2,4.2), 0.1, mix, color=colors[3],levels=[-0.025,-0.017,-0.01,0])
plot_level(25,nss,colors[0])
level_1 = nss.find('1')
classifier = lambda x: level_1.extra_info['classifier'].predict_single(x)
contour_plot((-4.2,4.2), (-4.2,4.2), 0.01, classifier,color='black', levels=None)
level_2 = nss.find('2')
classifier = lambda x: level_2.extra_info['classifier'].predict_single(x)
contour_plot((-4.2,0.4), (-4.2,4.2), 0.01, classifier,color='black', levels=None)
plt.xlabel(r'$x_1$', fontsize=15)
plt.ylabel(r'$x_2$', fontsize=15)
plt.tick_params(labelsize=15)

plt.savefig('examples/figures/images/mix_contour_nss.pdf',bbox_inches='tight')
plt.show()