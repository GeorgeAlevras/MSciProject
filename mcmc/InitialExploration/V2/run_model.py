from models import models
import numpy as np
from scipy.integrate import solve_ivp


def run_model(model, model_params, model_state_params):
    try:
        return run_seirv(models[model], model_params, model_state_params)
    except ValueError:
        print('Model not available')


def run_seirv(odes, model_params, model_state_params):
    '''
        model_params = (b, g, e, v)
        if odes have noise:
            model_params = (b, g, e, v, std_f)
        model_state_params = [population, infected, vac_frac, nat_imm_rate, days=50]
    '''
    population, infected, vac_frac, nat_imm_rate, days = model_state_params
    gamma = model_params[1]

    t_span = np.array([0, days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    susceptible = (population - infected - population*nat_imm_rate)*(1-vac_frac)
    exposed = 0.2*infected*(1-vac_frac)
    symptomatic = 0.8*infected*(1-vac_frac)
    recovered = population*nat_imm_rate
    x_0 = [susceptible, exposed, symptomatic, recovered, (population-susceptible-symptomatic-recovered)]
    
    solutions = solve_ivp(odes, t_span, x_0, args=model_params, t_eval=t)
    S = solutions.y[0]
    E = solutions.y[1]
    I = solutions.y[2]
    R = solutions.y[3]
    V = solutions.y[4]

    # Returns the growth rate as the R_t for all points in time
    R_t = [((1/gamma)*((I[i+1]-I[i])/(I[i]))+1) for i in range(len(I)-1)]
    
    return [S, E, I, R, V], max(R_t)