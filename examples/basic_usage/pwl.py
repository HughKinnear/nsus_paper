from nsus.implementation import SubsetSimulation, HillValleyGraphNSuS
from examples.numerical_examples.pwl.pwl import pwl
import nsus.plot as nplt
import matplotlib.pyplot as plt

sus = SubsetSimulation(performance_function=pwl,
                 level_probability=0.1,
                 level_size=500,
                 dimension=2,
                 seed=5,
                 threshold=0,
                 scale=1,
                 verbose=True)

sus.run()
nplt.all_levels(sus)
plt.show()

nss = HillValleyGraphNSuS(performance_function=pwl,
                           dimension=2,
                           level_size=500,
                           threshold=0,
                           level_probability=0.1,
                           seed=2,
                           graph_size=14,
                           max_branches=2,
                           scale=1,
                           verbose=True)

nss.run()
nplt.all_levels(nss)
plt.show()
