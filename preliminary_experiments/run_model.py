from models import models
import numpy as np
from scipy.integrate import solve_ivp

"""
    This file runs models based on the specified model.
"""


def run_model(model, population, R_0, vac_frac, nat_imm_rate, vac_ef, vaccination_rate, days=50, stochastic=False):
    if model == 'sirv':
        return run_sirv(models[model], population, R_0, vac_frac, nat_imm_rate, vac_ef, days, stochastic)
    elif model == 'sirv_c':
        return run_sirv_c(models[model], population, R_0, vac_frac, nat_imm_rate, vac_ef, vaccination_rate, days, stochastic)
    else:
        raise ValueError('Model not available')


def run_sirv(func, pop, R_0, vac_frac, nat_imm_rate, vac_ef, days, stochastic=False):
    t_span = np.array([0, days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    susceptible = (pop - 1e3 - pop*nat_imm_rate)*(1-vac_frac)
    infected = 1e3*(1-vac_frac)
    recovered = pop*nat_imm_rate
    x_0 = [susceptible, infected, recovered, (pop-susceptible-infected-recovered)]

    if stochastic:
        R_0 = np.random.normal(loc=4.5, scale=1)  # Draw an R_0 value from a Gaussian distribution
        g = np.random.normal(loc=(1/6), scale=(1/25))  # Draw an inverse serial number from a Gaussian distribution
        args = (R_0*g, g, vac_ef, pop)
    else:
        args = (R_0*(1/6), (1/6), vac_ef, pop)

    solutions = solve_ivp(func, t_span, x_0, args=args, t_eval=t)
    S = solutions.y[0]
    I = solutions.y[1]
    R = solutions.y[3]
    V = solutions.y[3]

    # Returns the growth rate as the R_t for all points in time
    R_t = [((1/args[1])*((I[i+1]-I[i])/(I[i]))+1) for i in range(len(I)-1)]
    
    return [S, I, R, V], max(R_t), args


def run_sirv_c(func, pop, R_0, vac_frac, nat_imm_rate, vac_ef, vaccination_rate, days, stochastic=False):
    t_span = np.array([0, days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    susceptible = (pop - 1e3 - pop*nat_imm_rate)*(1-vac_frac)
    symptomatic = 1e3*(1-vac_frac)
    recovered = pop*nat_imm_rate
    x_0 = [susceptible, 0, 0, symptomatic, 0, recovered, 0, (pop-susceptible-symptomatic-recovered), 0, 0, 0]

    if stochastic:
        g = (1/6)
        args = (R_0*g, R_0*g, vaccination_rate, 0.083333333, 0.15, 0.1, g, g, 0.075, 0.003428571, 0.025, (1-vac_ef)*R_0*g, (1-vac_ef)*R_0*g, 0.1, 0.00035, 0.1, 0.00125, 0.2, g)
        args = np.array(args)
        sigmas = 0.1*args
        args = [np.random.normal(m, s) for m, s in zip(args, sigmas)]
    else:
        args = (R_0*(1/6), R_0*(1/6), vaccination_rate, 0.083333333, 0.15, 0.1, (1/6), (1/6), 0.075, 0.003428571, 0.025, (1-vac_ef)*R_0*(1/6), (1-vac_ef)*R_0*(1/6), 0.1, 0.00035, 0.1, 0.00125, 0.2, (1/6))
    
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
    
    return [S, E, A, Sy, H, R, D, V, A_v, Sy_v, H_v], max(R_t), args
