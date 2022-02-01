import numpy as np


def seasyhrvd_odes(t, x, b_a, b_sy, g_a, g_sy, g_h, e_a, e_sy, a, h, v, d_h):    
    S = x[0]  # Susceptible
    E = x[1]  # Exposed
    A = x[2]  # Infected Asymptomatic
    Sy = x[3]  # Infected Symptomatic
    H = x[4]  # Hospitalised
    R = x[5]  # Recovered
    V = x[6]  # Vaccinated
    D = x[7]  # Deceased
    
    N = np.sum([S, E, A, Sy, R, H, V])  # Dynamic population (excludes deceased people)
    
    dSdt = -(b_a/N)*A*S -(b_sy/N)*Sy*S - v*S
    dEdt = (b_a/N)*A*S + (b_sy/N)*Sy*S - e_a*E - e_sy*E
    dAdt = e_a*E - g_a*A - a*A
    dSydt = e_sy*E + a*A - g_sy*Sy - h*Sy
    DHdt = h*Sy - g_h*Sy - d_h*H
    dRdt = g_a*A + g_sy*Sy + g_h*Sy
    dVdt = v*S
    dDdt = d_h*H

    return [dSdt, dEdt, dAdt, dSydt, DHdt, dRdt, dVdt, dDdt]


def seasyhrvd_odes_noise(t, x, b_a, b_sy, g_a, g_sy, g_h, e_a, e_sy, a, h, v, d_h, std_f=0):    
    S = x[0]  # Susceptible
    E = x[1]  # Exposed
    A = x[2]  # Infected Asymptomatic
    Sy = x[3]  # Infected Symptomatic
    H = x[4]  # Hospitalised
    R = x[5]  # Recovered
    V = x[6]  # Vaccinated
    D = x[7]  # Deceased

    N = np.sum([S, E, A, Sy, R, H, V])  # Dynamic population (excludes deceased people)
    
    b_a = abs(np.random.normal(b_a, std_f*b_a))
    b_sy = abs(np.random.normal(b_sy, std_f*b_sy))
    g_a = abs(np.random.normal(g_a, std_f*g_a))
    g_sy = abs(np.random.normal(g_sy, std_f*g_sy))
    g_h = abs(np.random.normal(g_h, std_f*g_h))
    e_a = abs(np.random.normal(e_a, std_f*e_a))
    e_sy = abs(np.random.normal(e_sy, std_f*e_sy))
    a = abs(np.random.normal(a, std_f*a))
    h = abs(np.random.normal(h, std_f*h))
    v = abs(np.random.normal(v, std_f*v))
    d_h = abs(np.random.normal(d_h, std_f*d_h))

    dSdt = -(b_a/N)*A*S -(b_sy/N)*Sy*S - v*S
    dEdt = (b_a/N)*A*S + (b_sy/N)*Sy*S - e_a*E - e_sy*E
    dAdt = e_a*E - g_a*A - a*A
    dSydt = e_sy*E + a*A - g_sy*Sy - h*Sy
    DHdt = h*Sy - g_h*Sy - d_h*H
    dRdt = g_a*A + g_sy*Sy + g_h*Sy
    dVdt = v*S
    dDdt = d_h*H

    return [dSdt, dEdt, dAdt, dSydt, DHdt, dRdt, dVdt, dDdt]


models = {
    'seasyhrvd':seasyhrvd_odes,
    'seasyhrvd_noise':seasyhrvd_odes_noise
}
