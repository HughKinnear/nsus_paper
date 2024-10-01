import numpy as np
from scipy.integrate import solve_ivp
    
class WhiteNoiseOscillator:

    def __init__(self,
                 dd_coeff,
                 d_coeff,
                 coeff,
                 duff_coeff,
                 sigma,
                 tau,
                 delta_t,
                 b,
                 initial_conditions,
                 call_type):
        self.dd_coeff = dd_coeff
        self.d_coeff = d_coeff
        self.coeff = coeff
        self.duff_coeff = duff_coeff
        self.sigma = sigma
        self.tau = tau
        self.delta_t = delta_t
        self.b = b
        self.initial_conditions = initial_conditions
        self.y_cache = {}
        self.time_steps = np.linspace(0, tau, int(tau  / delta_t) + 1)
        self.inv_dd_coeff = 1/dd_coeff
        self.call_type = call_type

    def out(self,x):
        self.scaled_parameters = self.sigma * np.array(x)
        out = solve_ivp(self.solver, (0,self.tau), self.initial_conditions)
        return out
    
    def y_no_cache(self,x):
        return self.out(x).y[0]

    def y(self,x):
        result = self.out(x)
        self.y_cache[tuple(x)] = result
        return result.y[0]
    
    def solver(self,t,y):
        return np.array([y[1], 
                        (-self.d_coeff*y[1]
                         -self.coeff*y[0]
                         -self.duff_coeff*y[0]**3
                         +self.interpolated(t))*self.inv_dd_coeff])
    
    def interpolated(self,t):
        return np.interp(t, self.time_steps, self.scaled_parameters)
    
    def top_final(self,x):
        return self.y(x)[-1] - self.b
    
    def top_any(self,x):
        return max(self.y(x)) - self.b
    
    def both_final(self,x):
        return abs(self.y(x)[-1]) - self.b
    
    def both_any(self,x):
        return max(abs(self.y(x))) - self.b
    
    def __call__(self,x):
        return getattr(self,self.call_type)(x)
    

class AuBeck(WhiteNoiseOscillator):

    def __init__(self,
                 b=2,
                 tau=30,
                 initial_conditions=(0,0),
                 S=1,
                 delta_t=0.02,
                 omega=7.85,
                 zeta=0.02,
                 call_type='both_any'
                 ):
        
        self.S = S
        self.omega = omega
        self.zeta = zeta

        super().__init__(dd_coeff=1,
                         d_coeff=2*zeta*omega,
                         coeff=omega**2,
                         duff_coeff=0,
                         sigma=np.sqrt(2*np.pi*S/delta_t),
                         tau=tau,
                         delta_t=delta_t,
                         b=b,
                         initial_conditions=initial_conditions,
                         call_type=call_type)
    
    def performance(self,x):
        return self.both_any(x)
        

def osc():
    return AuBeck(tau=1,b=1.2,delta_t=0.01,call_type='both_final')


def degenerate_osc(ss):
    osc = ss.performance_function.non_cache_performance_function
    all_end_points = np.array([osc.y_cache[tuple(samp.array)].y[0][-1] for samp in ss.all_samples])
    return not (any(all_end_points > 1.2) and any(all_end_points < -1.2))