import numpy as np
import pandas as pd
from copy import deepcopy
import lmfit
from scipy.integrate import odeint
from lmfit import Parameters
from utils import stepwise_soft, shift


#incluindo compartimento vacinacao

class SEIR:
    def __init__(self, stepwise_size=60, params=None):
        super().__init__()
        self.stepwise_size = stepwise_size
        self.params = params

    def get_fit_params(self, data):
        params = Parameters()
        params.add("population", value=12_000_000, vary=False)

        params.add("sigmoid_r", value=20, min=1, max=30, brute_step=1, vary=False)
        params.add("sigmoid_c", value=0.5, min=0, max=1, brute_step=0.1, vary=False)

        #fizemos epidemic_started_days_ago = 30 pois nosso dataset começa em 04.04.20 e o primeiro caso em RJ foi em 05.03.20
        params.add("epidemic_started_days_ago", value=30, min=1, max=90, brute_step=10, vary=False)

        #parametro inicial RJ encontrado na simulacao de 6meses: r0=3.5
        #params.add("r0", value=3.5, min=2, max=5, brute_step=0.05, vary=True)
        
        #parametro RJ usado na simulacao final completa: r0=2.5
        params.add("r0", value=2.5, min=2, max=3, brute_step=0.05, vary=True)

        #params.add("alpha", value=0.0064, min=0.005, max=0.0078, brute_step=0.0005, vary=True)  # CFR
        
        #parametro inicial encontrado na simulacao de 6meses: alpha=0.0078 e delta=0.5
        params.add("alpha", value=0.0078, min=0.005, max=0.0078, brute_step=0.0005, vary=True)
        params.add("delta", value=1/2, min=1/14, max=1/2, vary=True)  # E -> I rate
        
        #params.add("gamma", value=1/9, min=1/14, max=1/7, vary=False)  # I -> R rate
        params.add("gamma", value=1/14, min=1/14, max=1/7, vary=False)
        
    #vacinacao--------------------
        params.add("mi", value=0.4, min=0.2, max=1, brute_step=0.1, vary=False)
        params.add("lambda1", value=0.6, min=0.3, max=1, brute_step=0.1, vary=False)
        
    #-----------------------------    
        
        params.add("rho", expr='gamma', vary=False)  # I -> D rate

        params.add("incubation_days", expr='1/delta', vary=False)
        params.add("infectious_days", expr='1/gamma', vary=False)

        params.add(f"t0_q", value=0, min=0, max=0.99, brute_step=0.1, vary=False)
        piece_size = self.stepwise_size
        for t in range(piece_size, len(data), piece_size):
            params.add(f"t{t}_q", value=0.5, min=0, max=0.99, brute_step=0.1, vary=True)
        return params

    def get_initial_conditions(self, data):
        # Simulate such initial params as to obtain as many deaths as in data
        population = self.params['population']
        epidemic_started_days_ago = self.params['epidemic_started_days_ago']

        new_params = deepcopy(self.params)
        for key, value in new_params.items():
            if key.startswith('t'):
                new_params[key].value = 0
        new_model = self.__class__(params=new_params)

        t = np.arange(epidemic_started_days_ago)
        (S, E, I, R, D, V), history = new_model.predict(t, (population - 1, 0, 1, 0, 0, 0), history=False)

        I0 = I[-1]
        E0 = E[-1]
        Rec0 = R[-1]
        D0 = D[-1]
        S0 = S[-1]
        V0 = V[-1]
        return (S0, E0, I0, Rec0, D0, V0)

    def compute_daily_values(self, S, E, I, R, D, V):
        new_dead = (np.diff(D))
        new_recovered = (np.diff(R))
        new_infected = (np.diff(I))
        new_exposed = (np.diff(S[::-1])[::-1])
    #vacinacao---------    
        new_vaccinated = (np.diff(V))
        new_imunes = (np.diff(V))
        new_succetives = (np.diff(S))
    #------------------

        return new_exposed, new_infected, new_recovered, new_dead, new_vaccinated, new_succetives, new_imunes

    def get_step_rt_beta(self, t, params):
        r0 = params['r0']
        gamma = params['gamma']
        sigmoid_r = params['sigmoid_r']
        sigmoid_c = params['sigmoid_c']

        q_coefs = {}
        for key, value in params.items():
            if key.startswith('t'):
                coef_t = int(key.split('_')[0][1:])
                q_coefs[coef_t] = value.value

        quarantine_mult = stepwise_soft(t, q_coefs, r=sigmoid_r, c=sigmoid_c)
        rt = r0 - quarantine_mult * r0
        beta = rt * gamma
        return quarantine_mult, rt, beta

    def step(self, initial_conditions, t, params, history_store):
        population = params['population']
        delta = params['delta']
        gamma = params['gamma']
        alpha = params['alpha']
        rho = params['rho']
    #vacinacao---------------------
        mi = params['mi']
        lambda1 = params['lambda1'] 
    #------------------------------        
        

        quarantine_mult, rt, beta = self.get_step_rt_beta(t, params)

        S, E, I, R, D, V = initial_conditions

    #-----------------------------------------------
    #population= 12.000.000
    #beta = 2.5 * gamma = 2.5 * 0.071 = 0.1785
    #alpha = 0,032
    #gamma = 0.071
    #delta = 0,33
    #mi =
    #lambda1 =
    #r0 = 2.5
    
    
        #new_succetives = -beta * I * (S / population) - (0.2) * (S / population)
        #new_exposed = beta * I * (S / population) + beta * I * (V / population) - delta * E
        #new_infected = delta * E - gamma * (1 - alpha) * I
        #new_recovered = gamma * I + lambda1 * V
        #new_dead = gamma * alpha * I   
        
        new_succetives = -beta * I * (S / population)
        new_exposed = beta * I * (S / population) - delta * E
        #new_infected = delta * E
        new_infected = delta * E - gamma * alpha * I - gamma * (1 - alpha) * I
        new_recovered = gamma * (1 - alpha) * I 
        new_dead = gamma * alpha * I   
    
        
    #vacinacao--------------------------------------    
        #new_vaccinated = (0.2) * (S / population) - beta * I * (V / population) - lambda1 * V
        
        #new_vaccinated = (1 - beta) * I * (S / population)
        new_vaccinated = (1 - beta) * (S / population)
        new_imunes = beta * I * (V / population)
    #----------------------------------------------- 
        dSdt = new_succetives - new_vaccinated
        dEdt = new_exposed
        dIdt = new_infected
        dRdt = new_recovered
        dDdt = new_dead
    #vacinacao--------------------------------------
        dVdt = new_vaccinated
    #-----------------------------------------------

    
        
        

        assert S + E + I + R + D + V - population <= 1e10
        assert dSdt + dIdt + dEdt + dRdt + dDdt + dVdt <= 1e10

        if history_store is not None:
            history_record = {
                't': t+1,
                'quarantine_mult': quarantine_mult,
                'rt': rt,
                'beta': beta,
                'new_exposed': new_exposed,
                'new_infected': new_infected,
                'new_dead': new_dead,
                'new_recovered': new_recovered,
                'new_vaccinated': new_vaccinated,
                'new_succetives' : new_succetives,
            }
            history_store.append(history_record)

        return dSdt, dEdt, dIdt, dRdt, dDdt, dVdt

    def predict(self, t, initial_conditions, history=True):
        if history == True:
            history = []
        else:
            history = None

        ret = odeint(self.step, initial_conditions, t, args=(self.params, history))

        if history:
            history = pd.DataFrame(history)
            if not history.empty:
                history.index = history.t
                history = history[~history.index.duplicated(keep='first')]
        return ret.T, history


class SEIR_ID(SEIR):
    def get_fit_params(self, data):
        params = super().get_fit_params(data)

        #params.add("pi", value=0.2, min=0.15, max=0.3, brute_step=0.01, vary=True)  # Probability to discover a new infected case in a day
        #params.add("pd", value=0.35, min=0.15, max=0.9, brute_step=0.05, vary=True)  # Probability to discover a death
        
        #parametros inicias encontrados na simulacao de 6meses
        #params.add("pi", value=0.17, min=0.15, max=0.3, brute_step=0.01, vary=True)  # Probability to discover a new infected case in a day
        #params.add("pd", value=0.89, min=0.15, max=0.9, brute_step=0.05, vary=True)  # Probability to discover a death
        params.add("r0", value=2.5, min=2, max=3, brute_step=0.05, vary=True)
        
        params.add("pi", value=0.17, min=0.15, max=0.3, brute_step=0.01, vary=True)
        params.add("pd", value=0.79, min=0.15, max=0.9, brute_step=0.05, vary=True)
        
        
        return params

    def get_initial_conditions(self, data):
        population = self.params['population']
        epidemic_started_days_ago = self.params['epidemic_started_days_ago']

        new_params = deepcopy(self.params)
        for key, value in new_params.items():
            if key.startswith('t'):
                new_params[key].value = 0
        new_model = self.__class__(params=new_params)

        t = np.arange(epidemic_started_days_ago)
        (S, E, I, Iv, R, Rv, D, Dv, V), history = new_model.predict(t, (population-1, 0, 1, 0, 0, 0, 0, 0, 0), history=False)

        S0 = S[-1]
        E0 = E[-1]
        I0 = I[-1]
        Iv0 = Iv[-1]
        R0 = R[-1]
        Rv0 = Rv[-1]
        D0 = D[-1]
        Dv0 = Dv[-1]
        V0 = V[-1]
        return (S0, E0, I0, Iv0, R0, Rv0, D0, Dv0, V0)

    def step(self, initial_conditions, t, params, history_store):
        population = params['population']
        delta = params['delta']
        gamma = params['gamma']
        alpha = params['alpha']
        rho = params['rho']
        pi = params['pi']
        pd = params['pd']

        quarantine_mult, rt, beta = self.get_step_rt_beta(t, params)

        (S, E, I, Iv, R, Rv, D, Dv, V) = initial_conditions

        new_exposed = beta * (I+Iv) * (S / population)
        new_infected_inv = (1 - pi) * delta * E
        new_recovered_inv = gamma * (1 - alpha) * I
        new_dead_inv = (1 - pd) * alpha * rho * I
        new_dead_vis_from_I = pd * alpha * rho * I

        new_infected_vis = pi * delta * E
        new_recovered_vis = gamma * (1 - alpha) * Iv
        new_dead_vis_from_Iv = alpha * rho * Iv
        
        new_vaccinated = (1 - beta) * (S / population)

        dSdt = -new_exposed - new_vaccinated
        dEdt = new_exposed - new_infected_vis - new_infected_inv
        dIdt = new_infected_inv - new_recovered_inv - new_dead_inv - new_dead_vis_from_I
        dIvdt = new_infected_vis - new_recovered_vis - new_dead_vis_from_Iv
        dRdt = new_recovered_inv
        dRvdt = new_recovered_vis
        dDdt = new_dead_inv
        dDvdt = new_dead_vis_from_I + new_dead_vis_from_Iv
        
        dVdt = new_vaccinated

        assert S + E + I + Iv + R + Rv + D + Dv + V - population <= 1e10
        assert dSdt + dEdt + dIdt + dIvdt + dRdt + dRvdt + dDdt + dDvdt + dVdt <= 1e10

        if history_store is not None:
            history_record = {
                't': t+1,
                'quarantine_mult': quarantine_mult,
                'rt': rt,
                'beta': beta,
                'new_exposed': new_exposed,
                'new_infected_vis': new_infected_vis,
                'new_dead_vis': new_dead_vis_from_I + new_dead_vis_from_Iv,
                'new_recovered_vis': new_recovered_vis,
                'new_infected_inv': new_infected_inv,
                'new_dead_inv': new_dead_inv,
                'new_recovered_inv': new_recovered_inv,
                'new_vaccinated': new_vaccinated,
            }
            history_store.append(history_record)

        return dSdt, dEdt, dIdt, dIvdt, dRdt, dRvdt, dDdt, dDvdt, dVdt

    def compute_daily_values(self, S, E, I, Iv, R, Rv, D, Dv, V):
        new_dead_inv = (np.diff(D))
        new_recovered_inv = (np.diff(R))
        new_recovered_vis = (np.diff(Rv))
        new_exposed = (np.diff(S[::-1])[::-1])

        new_dead_vis_from_Iv = self.params['alpha'] * self.params['rho'] * (shift(Iv, 1)[1:])
        new_dead_vis_from_I = (np.diff(Dv)) - new_dead_vis_from_Iv
        new_dead_vis = new_dead_vis_from_Iv + new_dead_vis_from_I

        new_infected_vis = (np.diff(Iv)) + new_recovered_vis + new_dead_vis_from_Iv
        new_infected_inv = (np.diff(I)) + new_recovered_inv + new_dead_vis_from_I
          
        new_vaccinated = (np.diff(V))
                 
        return new_exposed, new_infected_inv, new_infected_vis, new_recovered_inv, new_recovered_vis, new_dead_inv, new_dead_vis, new_vaccinated

