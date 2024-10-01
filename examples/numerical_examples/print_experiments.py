from examples.numerical_examples.experiment import load_results

pwl_sus_exp = load_results('examples/numerical_examples/pwl/pwl_sus_exp.pkl')
pwl_nss_exp = load_results('examples/numerical_examples/pwl/pwl_nss_exp.pkl')
high_pwl_sus_exp = load_results('examples/numerical_examples/pwl/high_pwl_sus_exp.pkl')
high_pwl_nss_exp = load_results('examples/numerical_examples/pwl/high_pwl_nss_exp.pkl')
mix_sus_exp = load_results('examples/numerical_examples/mix/mix_sus_exp.pkl')
mix_nss_exp = load_results('examples/numerical_examples/mix/mix_nss_exp.pkl')
osc_sus_exp = load_results('examples/numerical_examples/osc/osc_sus_exp.pkl')
osc_nss_exp = load_results('examples/numerical_examples/osc/osc_nss_exp.pkl')
two_dof_sus_exp = load_results('examples/numerical_examples/two_dof/two_dof_sus_exp.pkl')
two_dof_nss_exp = load_results('examples/numerical_examples/two_dof/two_dof_nss_exp.pkl')


def print_divider():
    print('\n')
    print('='*80)
    print('='*80)
    print('\n')

print_divider()
pwl_sus_exp[1].print_table()

print_divider()
pwl_nss_exp[3].print_table()

print_divider()
high_pwl_sus_exp[1].print_table()

print_divider()
high_pwl_nss_exp[3].print_table()

print_divider()
mix_sus_exp[0].print_table()

print_divider()
mix_nss_exp[0].print_table()

print_divider()
osc_sus_exp[0].print_table()

print_divider()
osc_nss_exp[0].print_table()

print_divider()
two_dof_sus_exp[0].print_table()

print_divider()
two_dof_nss_exp[0].print_table()
