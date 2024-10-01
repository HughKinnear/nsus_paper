import pickle
from examples.numerical_examples.direct_monete_carlo import DirectMonteCarlo
from examples.numerical_examples.pwl.pwl import pwl

dmc = DirectMonteCarlo(dim=2,
                       performance_function=pwl,
                       sample_size=10**8,
                       chunks=100,
                       seed=0,
                       is_sample_save=False)

dmc.compute()

estimate = dmc.threshold(0)

with open('pwl_ref_prob.pkl', 'wb') as file:
    pickle.dump(estimate, file)



