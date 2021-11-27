from noisy_models import noisy_models
import numpy as np
from scipy.integrate import solve_ivp

"""
    This file runs noisy models based on the specified model.
"""

def run_noisy_model(model, population, R_0, vac_frac, nat_imm_rate, vac_ef, vaccination_rate, days=50, stochastic=False, std_f=0.05):
    if model == 'sirv':
        return run_sirv(noisy_models[model], population, R_0, vac_frac, nat_imm_rate, vac_ef, days, stochastic, std_f)
    elif model == 'sirv_c':
        return run_sirv_c(noisy_models[model], population, R_0, vac_frac, nat_imm_rate, vac_ef, vaccination_rate, days, stochastic, std_f)
    else:
        raise ValueError('Model not available')


def run_sirv(func, pop, R_0, vac_frac, nat_imm_rate, vac_ef, days, stochastic=False, std_f=0.05):
    t_span = np.array([0, days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    susceptible = (pop - 1e3 - pop*nat_imm_rate)*(1-vac_frac)
    infected = 1e3*(1-vac_frac)
    recovered = pop*nat_imm_rate
    x_0 = [susceptible, infected, recovered, (pop-susceptible-infected-recovered)]
    args = (R_0*(1/6), (1/6), vac_ef, pop, std_f)

    solutions = solve_ivp(func, t_span, x_0, args=args, t_eval=t)
    S = solutions.y[0]
    I = solutions.y[1]
    R = solutions.y[3]
    V = solutions.y[3]

    # Returns the growth rate as the R_t for all points in time
    R_t = [((1/args[1])*((I[i+1]-I[i])/(I[i]))+1) for i in range(len(I)-1)]
    
    return [S, I, R, V], max(R_t)


def run_sirv_c(func, pop, R_0, vac_frac, nat_imm_rate, vac_ef, vaccination_rate, days, stochastic=False, std_f=0.05):
    t_span = np.array([0, days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    susceptible = (pop - 1e3 - pop*nat_imm_rate)*(1-vac_frac)
    symptomatic = 1e3*(1-vac_frac)
    recovered = pop*nat_imm_rate
    x_0 = [susceptible, 0, 0, symptomatic, 0, recovered, 0, (pop-susceptible-symptomatic-recovered), 0, 0, 0]
    args = (R_0*(1/6), R_0*(1/6), vaccination_rate, 0.083333333, 0.15, 0.1, (1/6), (1/6), 0.075, 0.003428571, 0.025, (1-vac_ef)*R_0*(1/6), (1-vac_ef)*R_0*(1/6), 0.1, 0.00035, 0.1, 0.00125, 0.2, (1/6), std_f)
    
    solutions = solve_ivp(func, t_span, x_0, args=args, t_eval=t)
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
    R_t = [((1/args[6])*((A[i+1]+Sy[i+1]+A_v[i+1]+Sy_v[i+1]-A[i]-Sy[i]-A_v[i]-Sy_v[i])/(A[i]+Sy[i]+A_v[i]+Sy_v[i]))+1) for i in range(len(Sy)-1)]
    
    return [S, E, A, Sy, H, R, D, V, A_v, Sy_v, H_v], max(R_t)
