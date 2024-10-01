from nsus.implementation import SubsetSimulation
from nsus.estimate import exceedance_probability
from examples.numerical_examples.osc.osc import osc
import pickle


osc_func = osc()

sus = SubsetSimulation(performance_function=osc_func,
                       dimension=101,
                       level_size=10**5,
                       threshold=0,
                       level_probability=0.1,
                       seed=2,
                       scale=1,
                       verbose=True)

sus.run()
estimate = exceedance_probability(sus,0)

with open('osc_ref_prob.pkl', 'wb') as file:
    pickle.dump(estimate, file)