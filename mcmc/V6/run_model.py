from email.header import decode_header
from models import models
import numpy as np
from scipy.integrate import solve_ivp


def run_model(model, model_params, model_state_params):
    try:
        return run_seasyhrva_vsy_vh_vd(models[model], model_params, model_state_params)
    except ValueError:
        print('Model not available')


def run_seasyhrva_vsy_vh_vd(odes, model_params, model_state_params):
    '''
        model_params = (b_a, b_sy, g_a, g_sy, g_h, e_a, e_sy, a, h, v, b_a_v, b_sy_v, g_a_v, g_sy_v, g_h_v, a_v, h_v, d_h, d_h_v)
        if odes have noise:
            model_params = (b_a, b_sy, g_a, g_sy, g_h, e_a, e_sy, a, h, v, b_a_v, b_sy_v, g_a_v, g_sy_v, g_h_v, a_v, h_v, d_h, d_h_v, std_f)
        model_state_params = [population, infected, vac_frac, nat_imm_rate, days=50]
    '''
    population, infected, vac_frac, nat_imm_rate, days = model_state_params
    gamma = model_params[1]

    t_span = np.array([0, days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    susceptible = (population - infected - population*nat_imm_rate)*(1-vac_frac)
    exposed = 0.2*infected*(1-vac_frac)
    asymptomatic = 0.2*0.8*infected*(1-vac_frac)
    symptomatic = 0.6*0.8*infected*(1-vac_frac)
    asymptomatic_vac = 0.1*infected*(1-vac_frac)
    symptomatic_vac = 0.1*infected*(1-vac_frac)
    hospitalised = 0
    hospitalised_vac = 0
    recovered = population*nat_imm_rate
    deceased = 0
    x_0 = [susceptible, exposed, asymptomatic, symptomatic, asymptomatic_vac, symptomatic_vac, hospitalised, hospitalised_vac, \
         recovered, (population-susceptible-symptomatic-recovered), deceased]
    
    solutions = solve_ivp(odes, t_span, x_0, args=model_params, t_eval=t)
    S = solutions.y[0]
    E = solutions.y[1]
    A = solutions.y[2]
    Sy = solutions.y[3]
    A_v = solutions.y[4]
    Sy_v = solutions.y[5]
    H = solutions.y[6]
    H_v = solutions.y[7]
    R = solutions.y[8]
    V = solutions.y[9]
    D = solutions.y[10]

    # Returns the growth rate as the R_t for all points in time
    R_t = [((1/gamma)*((A[i+1]+Sy[i+1]+A_v[i+1]+Sy_v[i+1]-A[i]-Sy[i]-A_v[i]-Sy_v[i])/(A[i]+Sy[i]+A_v[i]+Sy_v[i]))+1) for i in range(len(A)-1)]
    
    return [S, E, A, Sy, A_v, Sy_v, H, H_v, R, V, D], max(R_t)
