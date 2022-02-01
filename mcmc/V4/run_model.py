from email.header import decode_header
from models import models
import numpy as np
from scipy.integrate import solve_ivp


def run_model(model, model_params, model_state_params):
    try:
        return run_seasyrvd(models[model], model_params, model_state_params)
    except ValueError:
        print('Model not available')


def run_seasyrvd(odes, model_params, model_state_params):
    '''
        model_params = (b_a, b_sy, g_a, g_sy, e_a, e_sy, a, v, d)
        if odes have noise:
            model_params = (b_a, b_sy, g_a, g_sy, e_a, e_sy, a, v, d, std_f)
        model_state_params = [population, infected, vac_frac, nat_imm_rate, days=50]
    '''
    population, infected, vac_frac, nat_imm_rate, days = model_state_params
    gamma = model_params[1]

    t_span = np.array([0, days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    susceptible = (population - infected - population*nat_imm_rate)*(1-vac_frac)
    exposed = 0.2*infected*(1-vac_frac)
    asymptomatic = 0.3*0.8*infected*(1-vac_frac)
    symptomatic = 0.7*0.8*infected*(1-vac_frac)
    recovered = population*nat_imm_rate
    deceased = 0
    x_0 = [susceptible, exposed, asymptomatic, symptomatic, recovered, (population-susceptible-symptomatic-recovered), deceased]
    
    solutions = solve_ivp(odes, t_span, x_0, args=model_params, t_eval=t)
    S = solutions.y[0]
    E = solutions.y[1]
    A = solutions.y[2]
    Sy = solutions.y[3]
    R = solutions.y[4]
    V = solutions.y[5]
    D = solutions.y[6]

    # Returns the growth rate as the R_t for all points in time
    R_t = [((1/gamma)*((A[i+1]+Sy[i+1]-A[i]-Sy[i])/(A[i]+Sy[i]))+1) for i in range(len(A)-1)]
    
    return [S, E, A, Sy, R, V, D], max(R_t)
