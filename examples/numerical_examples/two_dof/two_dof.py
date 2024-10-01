import numpy as np
from scipy.linalg import eigh


def two_dof(x):
    
    m1 = 2000
    m2 = 2000
    M = np.diag([m1, m2])
    sigma = np.sqrt(np.log(1.04))
    mu = np.log(2.5*10**5) - 0.5 * sigma**2
    x = np.array(x)
    k = np.exp(mu + sigma * x)
    k1 = k[0]
    k2 = k[1]
    K = np.array([[k1 + k2, -k2], [-k2, k2]])
    OME = 11
    tm = np.arange(0, 20.01, 0.01)
    
    evals, evecs = eigh(K, M)
    nat_freq = np.sqrt(evals)
    sort_ind = np.argsort(nat_freq)
    nat_freq = nat_freq[sort_ind]
    evecs = evecs[:, sort_ind]

    mass_norm_evec = evecs / np.sqrt(np.diag(evecs.T @ M @ evecs))

    mod_resp = np.full((len(tm), 2), np.nan)
    
    for i in range(2):
        w = nat_freq[i]
        r = OME / w
        eta = 0.02
        tanth = 2 * eta * r / (1 - r**2)
        sinth = np.sqrt(1- 1 / (1 + tanth**2))
        costh = np.sign(tanth) * np.sqrt(1 / (1 + tanth**2))
        wd = w * np.sqrt(1 - eta**2)
        A2 = sinth * (2000 * mass_norm_evec[1, i]) / (w**2 * np.sqrt((1 - r**2)**2 + (2 * eta * r)**2))
        A1 = 1 / wd * (A2 * eta * w - (2000 * mass_norm_evec[1, i]) * OME * costh / 
                       (w**2 * np.sqrt((1 - r**2)**2 + (2 * eta * r)**2)))
        p = np.exp(-eta * w * tm) * (A1 * np.sin(wd * tm) + A2 * np.cos(wd * tm)) + \
            (2000 * mass_norm_evec[1, i]) * (np.sin(OME * tm) * costh - np.cos(OME * tm) * sinth) / \
            (w**2 * np.sqrt((1 - r**2)**2 + (2 * eta * r)**2))
        mod_resp[:, i] = np.real(p)
    mass_resp = mass_norm_evec @ mod_resp.T
    y = np.max(mass_resp[0, :]) - 0.024
    return y


def degenerate_two_dof(ss):
    arrays = np.array([sample.array for sample in ss.all_samples])
    pos_bool = arrays[:,0] < 0
    perf_array = np.array([sample.performance for sample in ss.all_samples])
    perf_bool = perf_array > 0
    return not (pos_bool & perf_bool).any()