import numpy as np
from scipy.stats import norm, uniform


class ChainData:

    def __init__(self,
                 chain_list,
                 parameter_names=None):
        self.chain_list = chain_list

        if parameter_names is None:
            self.parameter_names = ['Parameter ' + str(i + 1)
                                    for i in range(self.number_of_params)]
        else:
            self.parameter_names = parameter_names

    @property
    def number_of_params(self):
        return len(self.chain_list[0][0])
    
    @property
    def chain_number(self):
        return len(self.chain_list)

    def trim(self, length):
        self.chain_list = [chain[length:] for chain in self.chain_list]

    @property
    def chain_dict(self):
        array_chain_list = [[sample for sample in chain]
                            for chain in self.chain_list]
        parameter_chain_list = [arr.T for arr in np.array(array_chain_list).T]
        chain_dict = {}
        for i in range(self.number_of_params):
            chain_dict[self.parameter_names[i]] = parameter_chain_list[i]
        return chain_dict

    @property
    def all_samples(self):
        return [item for chain in self.chain_list for item in chain]
    

class ModifiedMetropolis:
    
    def __init__(self,
                 indicator,
                 scale,
                 random_state):
        self.indicator = indicator
        self.scale = scale
        self.random_state = random_state

    def update(self, chain_data, length):
        sample_shape = (chain_data.chain_number,
                        chain_data.number_of_params,
                        length)
        loguniform_samps = np.log(uniform.rvs(size=sample_shape,random_state=self.random_state))
        conversion = self.scale_convert(chain_data.chain_number,length)
        proposal_samps = norm.rvs(size=sample_shape,scale=conversion,random_state=self.random_state)
        state_list = [np.array([chain[-1] for chain in chain_data.chain_list])]
        for i in range(length):
            current_state = state_list[-1]
            prop_state = current_state + proposal_samps[:,:,i]
            alpha = norm.logpdf(prop_state) - norm.logpdf(current_state)
            accept = alpha >= loguniform_samps[:,:,i]
            new_state = np.where(accept, prop_state, current_state)
            ind_accept = np.array([[bool(self.indicator(state))] for state in new_state])
            state_list.append(np.where(ind_accept, new_state, current_state))
        for state in state_list[1:]:
            for sample,chain in zip(state,chain_data.chain_list):
                chain.append(sample)

    def scale_convert(self, chain_number, length):
        reshaped_scales = self.scale.reshape(1,self.scale.shape[0],1)
        expanded_array = np.tile(reshaped_scales, (chain_number, 1, length))
        return expanded_array



        

    



