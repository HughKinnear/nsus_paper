from examples.numerical_examples.experiment import load_results
import matplotlib.pyplot as plt

colors = [
    '#e66101',
    '#fdb863',
    '#b2abd2',
    '#5e3c99'
]

pwl_sus_exp = load_results('examples/numerical_examples/pwl/pwl_sus_exp.pkl')
pwl_nss_exp = load_results('examples/numerical_examples/pwl/pwl_nss_exp.pkl')
high_pwl_sus_exp = load_results('examples/numerical_examples/pwl/high_pwl_sus_exp.pkl')
high_pwl_nss_exp = load_results('examples/numerical_examples/pwl/high_pwl_nss_exp.pkl')

sus_perf_evals = [exp.joint_eval_mean for exp in pwl_sus_exp]
sus_non_degeneracy = [exp.non_degen_percentage for exp in pwl_sus_exp]
nss_15_perf_evals = [exp.joint_eval_mean for exp in pwl_nss_exp if exp.graph_size == 15]
nss_20_perf_evals = [exp.joint_eval_mean for exp in pwl_nss_exp if exp.graph_size == 20]
nss_25_perf_evals = [exp.joint_eval_mean for exp in pwl_nss_exp if exp.graph_size == 25]
nss_15_non_degeneracy = [exp.non_degen_percentage 
                         for exp in pwl_nss_exp
                         if exp.graph_size == 15]
nss_20_non_degeneracy = [exp.non_degen_percentage
                         for exp in pwl_nss_exp
                         if exp.graph_size == 20]
nss_25_non_degeneracy = [exp.non_degen_percentage
                         for exp in pwl_nss_exp
                         if exp.graph_size == 25]

high_sus_perf_evals = [exp.joint_eval_mean for exp in high_pwl_sus_exp]
high_sus_non_degeneracy = [exp.non_degen_percentage for exp in high_pwl_sus_exp]
high_nss_15_perf_evals = [exp.joint_eval_mean for exp in high_pwl_nss_exp if exp.graph_size == 15]
high_nss_20_perf_evals = [exp.joint_eval_mean for exp in high_pwl_nss_exp if exp.graph_size == 20]
high_nss_25_perf_evals = [exp.joint_eval_mean for exp in high_pwl_nss_exp if exp.graph_size == 25]
high_nss_15_non_degeneracy = [exp.non_degen_percentage 
                         for exp in high_pwl_nss_exp
                         if exp.graph_size == 15]
high_nss_20_non_degeneracy = [exp.non_degen_percentage
                         for exp in high_pwl_nss_exp
                         if exp.graph_size == 20]
high_nss_25_non_degeneracy = [exp.non_degen_percentage
                         for exp in high_pwl_nss_exp
                         if exp.graph_size == 25]


plt.style.use('ggplot')
plt.figure(figsize=(10, 7))
plt.plot(sus_perf_evals,
         sus_non_degeneracy,
         marker='o',
         markersize=4,
         color=colors[3],
         label='SuS')

plt.plot(nss_15_perf_evals,
         nss_15_non_degeneracy,
         marker='o',
         markersize=4,
         linestyle='--', 
         color=colors[0],
         label='NSuS - Graph Size 15')

plt.plot(nss_20_perf_evals,
         nss_20_non_degeneracy,
         marker='o',
         markersize=4,
         linestyle=':', 
         color=colors[0],
         label='NSuS - Graph Size 20')

plt.plot(nss_25_perf_evals,
         nss_25_non_degeneracy,
         marker='o',
         markersize=4,
         linestyle='-', 
         color=colors[0],
         label='NSuS - Graph Size 25')


plt.xlabel('Performance Evaluations',fontsize=20)
plt.ylabel('Non-Degeneracy Percentage',fontsize=20)
plt.tick_params(labelsize=18)
plt.ticklabel_format(axis='x', style='sci', scilimits=(3,3), useMathText=True)
# plt.legend(markerscale=0,facecolor='white')
ax = plt.gca()
ax.xaxis.get_offset_text().set_size(15)
plt.savefig('examples/figures/images/pwl_degen.pdf',bbox_inches='tight')
plt.show()

plt.style.use('ggplot')
plt.figure(figsize=(10, 7))
ax = plt.gca()
plt.plot(high_sus_perf_evals,
         high_sus_non_degeneracy,
         marker='o',
         markersize=4,
         color=colors[3],
         label='SuS')


plt.plot(high_nss_15_perf_evals,
         high_nss_15_non_degeneracy,
         marker='o',
         markersize=4,
         linestyle='--', 
         color=colors[0],
         label='NSuS - Graph Size 15')

plt.plot(high_nss_20_perf_evals,
         high_nss_20_non_degeneracy,
         marker='o',
         markersize=4,
         linestyle=':', 
         color=colors[0],
         label='NSuS - Graph Size 20')

plt.plot(high_nss_25_perf_evals,
         high_nss_25_non_degeneracy,
         marker='o',
         markersize=4,
         linestyle='-', 
         color=colors[0],
         label='NSuS - Graph Size 25')


plt.xlabel('Performance Evaluations',fontsize=20)
plt.ylabel('Non-Degeneracy Percentage',fontsize=20)
plt.tick_params(labelsize=18)
plt.ticklabel_format(axis='x', style='sci', scilimits=(3,3), useMathText=True)
# plt.legend(markerscale=0,facecolor='white')
ax = plt.gca()
ax.xaxis.get_offset_text().set_size(15)
plt.savefig('examples/figures/images/high_pwl_degen.pdf',bbox_inches='tight')
plt.show()

#############

plt.style.use('default')
plt.figure(figsize=(1,1))
plt.axis('off')
plt.legend(*ax.get_legend_handles_labels(),ncols=2)
plt.savefig('examples/figures/images/pwl_degen_legend.pdf',bbox_inches='tight')
plt.show()



