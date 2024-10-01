import matplotlib.pyplot as plt
from nsus.estimate import failure_to_reliability
import seaborn as sns
from examples.numerical_examples.experiment import load_results

pwl_sus_exp = load_results('examples/numerical_examples/pwl/pwl_sus_exp.pkl')
pwl_nss_exp = load_results('examples/numerical_examples/pwl/pwl_nss_exp.pkl')
pwl_sus_exp[1].print_table()
sus_probs = pwl_sus_exp[1].joint_probs
nss_probs = pwl_nss_exp[3].joint_probs
ref_prob = pwl_sus_exp[3].ref_prob

colors = [
    '#e66101',
    '#fdb863',
    '#b2abd2',
    '#5e3c99'
]

plt.style.use('ggplot')

plt.figure(figsize=(10, 7))

sns.kdeplot(data=sus_probs,
            fill=True,
            linewidth=1,
            log_scale=True,
            color=colors[3],
            )

plt.axvline(ref_prob,
            color='black',
            ymax=1,
            alpha=1,
            linestyle='dashed',
            linewidth=1,
            label='Ref Probability')

plt.xlabel('Failure Probability Estimate',fontsize=15)
plt.ylabel('Denstiy',fontsize=15)
plt.minorticks_off()
plt.tick_params(labelsize=15)
plt.legend(fontsize=15,frameon=True,facecolor='white',loc='upper right')
plt.savefig('examples/figures/images/sus_kde.pdf',bbox_inches='tight')

plt.show()


plt.style.use('ggplot')

plt.figure(figsize=(10, 7))

sns.kdeplot(data=sus_probs,
            fill=True,
            linewidth=1,
            color=colors[3],
            log_scale=True,
            label='SuS'
            )

sns.kdeplot(data=nss_probs,
            fill=True,
            linewidth=1,
            color=colors[0],
            log_scale=True,
            label='NSuS'
            )


plt.axvline(ref_prob,
            color='black',
            ymax=1,
            alpha=1,
            linestyle='dashed',
            linewidth=1,
            label='Ref Probability')

plt.xlabel('Failure Probability Estimate',fontsize=15)
plt.ylabel('Denstiy',fontsize=15)
plt.tick_params(labelsize=15)
plt.minorticks_off()

plt.legend(fontsize=15,frameon=True,facecolor='white',loc='upper left')
plt.savefig('examples/figures/images/nss_kde.pdf',bbox_inches='tight')

plt.show()


