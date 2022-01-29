from models import models
import numpy as np
from scipy.integrate import solve_ivp


def run_model(model, model_params, model_state_params):
    try:
        if 'sirv' in model:
            return run_sirv(models[model], model_params, model_state_params)
    except ValueError:
        print('Model not available')


def run_sirv(odes, model_params, model_state_params):
    '''
        model_params = (b, g, v)
        if odes have noise:
            model_params = (b, g, v, std_f)
        model_state_params = [population, infected, vac_frac, nat_imm_rate, days=50]
    '''
    population, infected, vac_frac, nat_imm_rate, days = model_state_params
    gamma = model_params[1]

    t_span = np.array([0, days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    susceptible = (population - infected - population*nat_imm_rate)*(1-vac_frac)
    symptomatic = infected*(1-vac_frac)
    recovered = population*nat_imm_rate
    x_0 = [susceptible, symptomatic, recovered, (population-susceptible-symptomatic-recovered)]
    
    solutions = solve_ivp(odes, t_span, x_0, args=model_params, t_eval=t)
    S = solutions.y[0]
    I = solutions.y[1]
    R = solutions.y[2]
    V = solutions.y[3]

    # Returns the growth rate as the R_t for all points in time
    R_t = [((1/gamma)*((I[i+1]-I[i])/(I[i]))+1) for i in range(len(I)-1)]
    
    return [S, I, R, V], max(R_t)
