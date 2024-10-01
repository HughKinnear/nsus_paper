# Niching Subset Simulation

This is a companion repository for the paper Niching subset simulation. It has two purposes.

The first is reproducibility. All the code for the numerical experiments (examples/numerical_examples) and figures (examples/figures) used in the paper are shared here. The file examples/numerical_examples/print_experiments prints the results of all numerical experiments conducted.

The second is to provide an implementation of Niching Subset Simulation that others can use on their own problems. The file nsus/nsus.py contains the class NSuS that can be used to create an implementation of the Niching Subset Simulation Framework. The file nsus/implementation.py shows two examples of how to do this. The file examples/basic_usage/pwl.py shows how to use those implementations.