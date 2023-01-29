from models import models
import numpy as np
from scipy.integrate import solve_ivp


def run_model(model, model_params, model_state_params):
    try:
        return run_sirv_c(models[model], model_params, model_state_params)
    except ValueError:
        print('Model not available')


def run_sirv_c(odes, model_params, model_state_params):
    '''
        model_params = (b_a, b_sy, v, e_a, e_sy, a, g_a, g_sy, g_h, h, d_h, b_a_v, b_sy_v, a_v, h_v, g_h_v, d_h_v, g_a_v, g_sy_v)
        if odes have noise:
            model_params = (b_a, b_sy, v, e_a, e_sy, a, g_a, g_sy, g_h, h, d_h, b_a_v, b_sy_v, a_v, h_v, g_h_v, d_h_v, g_a_v, g_sy_v, std_f)
        model_state_params = [population, infected, vac_frac, nat_imm_rate, days=50]
    '''
    population, infected, vac_frac, nat_imm_rate, days = model_state_params
    gamma = model_params[6]

    t_span = np.array([0, days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    susceptible = (population - infected - population*nat_imm_rate)*(1-vac_frac)
    symptomatic = infected*(1-vac_frac)
    recovered = population*nat_imm_rate
    x_0 = [susceptible, 0, 0, symptomatic, 0, recovered, 0, (population-susceptible-symptomatic-recovered), 0, 0, 0]
    
    solutions = solve_ivp(odes, t_span, x_0, args=model_params, t_eval=t)
    S = solutions.y[0]
    E = solutions.y[1]
    A = solutions.y[2]
    Sy = solutions.y[3]
    H = solutions.y[4]
    R = solutions.y[5]
    D = solutions.y[6]
    V = solutions.y[7]
    A_v = solutions.y[8]
    Sy_v = solutions.y[9]
    H_v = solutions.y[10]

    # Returns the growth rate as the R_t for all points in time
    R_t = [((1/gamma)*((A[i+1]+Sy[i+1]+A_v[i+1]+Sy_v[i+1]-A[i]-Sy[i]-A_v[i]-Sy_v[i])/(A[i]+Sy[i]+A_v[i]+Sy_v[i]))+1) for i in range(len(Sy)-1)]
    
    return [S, E, A, Sy, H, R, D, V, A_v, Sy_v, H_v], max(R_t)
