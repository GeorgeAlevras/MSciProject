import numpy as np


def odes(t, x, b_a, b_sy, v, e_a, e_sy, a, g_a, g_sy, g_h, h, d_h, b_a_v, b_sy_v, a_v, h_v, g_h_v, d_h_v, g_a_v, g_sy_v):    
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


def odes_noise(t, x, b_a, b_sy, v, e_a, e_sy, a, g_a, g_sy, g_h, h, d_h, b_a_v, b_sy_v, a_v, h_v, g_h_v, d_h_v, g_a_v, g_sy_v, std_f=0):    
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
    
    b_a = abs(np.random.normal(b_a, std_f*b_a))
    b_sy = abs(np.random.normal(b_sy, std_f*b_sy))
    v = abs(np.random.normal(v, std_f*v))
    e_a = abs(np.random.normal(e_a, std_f*e_a))
    e_sy = abs(np.random.normal(e_sy, std_f*e_sy))
    a = abs(np.random.normal(a, std_f*a))
    g_a = abs(np.random.normal(g_a, std_f*g_a))
    g_sy = abs(np.random.normal(g_sy, std_f*g_sy))
    g_h = abs(np.random.normal(g_h, std_f*g_h))
    h = abs(np.random.normal(h, std_f*h))
    d_h = abs(np.random.normal(d_h, std_f*d_h))
    b_a_v = abs(np.random.normal(b_a_v, std_f*b_a_v))
    b_sy_v = abs(np.random.normal(b_sy_v, std_f*b_sy_v))
    a_v = abs(np.random.normal(a_v, std_f*a_v))
    h_v = abs(np.random.normal(h_v, std_f*h_v))
    g_h_v = abs(np.random.normal(g_h_v, std_f*g_h_v))
    d_h_v = abs(np.random.normal(d_h_v, std_f*d_h_v))
    g_a_v = abs(np.random.normal(g_a_v, std_f*g_a_v))
    g_sy_v = abs(np.random.normal(g_sy_v, std_f*g_sy_v))

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


models = {
    'sirv_c':odes,
    'sirv_c_noise':odes_noise
}