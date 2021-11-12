import numpy as np

"""
    This file contains all the compartmental models - their ODEs

"""


def sirv_odes(t, x, b, g, v_e, N):
    S = x[0]  # Susceptible
    I = x[1]  # Infected
    R = x[2]  # Recovered
    V = x[3]  # Vaccinated

    dSdt = -b*(I/N)*S
    dIdt = b*(I/N)*S - g*I + b*(1-v_e)*(I/N)*V
    dRdt = g*I
    dVdt = -b*(1-v_e)*(I/N)*V

    return [dSdt, dIdt, dRdt, dVdt]


def sirv_c_odes(t, x, b_a, b_sy, v, e_a, e_sy, a, g_a, g_sy, g_h, h, d_h, b_a_v, b_sy_v, a_v, h_v, g_h_v, d_h_v, g_a_v, g_sy_v):    
    S = x[0]  # Susceptible
    E = x[1]  # Exposed
    A = x[2]  # Asymptomatic
    Sy = x[3]  # Symptomatic
    H = x[4]  # Hospitalised
    R = x[5]  # Recovered
    D = x[6]  # Deceased
    V = x[7]  # Vaccinated
    A_v = x[8]  # Asymptomatic (vaccinated)
    Sy_v = x[9]  # Symptomatic (vaccinated)
    H_v = x[10]  # Hospitalised (vaccinated)
    
    N = np.sum([S, E, A, Sy, H, R, V, A_v, Sy_v, H_v])  # Dynamic population (excludes deceased people)
    
    dSdt = - b_a*((A+A_v)/N)*S - b_sy*((Sy+Sy_v)/N)*S - v*S
    dEdt = b_a*((A+A_v)/N)*S + b_sy*((Sy+Sy_v)/N)*S - e_a*E - e_sy*E
    dAdt = e_a*E - a*A - g_a*A
    dSydt = e_sy*E + a*A - g_sy*Sy - h*Sy
    dHdt = h*Sy - g_h*H - d_h*H
    dRdt = g_a*A + g_sy*Sy + g_h*H + g_h_v*H_v + g_a_v*A_v + g_sy_v*Sy_v
    dDdt = d_h*H + d_h_v*H_v
    dVdt = v*S - b_a_v*((A+A_v)*V)/N - b_sy_v*((Sy+Sy_v)*V)/N
    dA_vdt = b_a_v*((A+A_v)*V)/N - a_v*A_v - g_a_v*A_v
    dSy_vdt = b_sy_v*((Sy+Sy_v)*V)/N + a_v*A_v - h_v*Sy_v - g_sy_v*Sy_v
    dH_vdt = h_v*Sy_v - g_h_v*H_v - d_h_v*H_v

    return [dSdt, dEdt, dAdt, dSydt, dHdt, dRdt, dDdt, dVdt, dA_vdt, dSy_vdt, dH_vdt]


# This dictionary stores the function name for each model
models = {
    'sirv': sirv_odes,  # simple SIRV model
    'sirv_c': sirv_c_odes  # SIRV model with separate paths for vaccinated people
}
