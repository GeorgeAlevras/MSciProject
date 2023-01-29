import numpy as np


def seasyrvd_odes(t, x, b_a, b_sy, g_a, g_sy, e_a, e_sy, a, v, d):    
    S = x[0]  # Susceptible
    E = x[1]  # Exposed
    A = x[2]  # Infected Asymptomatic
    Sy = x[3]  # Infected Symptomatic
    R = x[4]  # Recovered
    V = x[5]  # Vaccinated
    D = x[6]  # Deceased
    
    N = np.sum([S, E, A, Sy, R, V])  # Dynamic population (excludes deceased people)
    
    dSdt = -(b_a/N)*A*S -(b_sy/N)*Sy*S - v*S
    dEdt = (b_a/N)*A*S + (b_sy/N)*Sy*S - e_a*E - e_sy*E
    dAdt = e_a*E - g_a*A - a*A
    dSydt = e_sy*E + a*A - g_sy*Sy - d*Sy
    dRdt = g_a*A + g_sy*Sy
    dVdt = v*S
    dDdt = d*Sy

    return [dSdt, dEdt, dAdt, dSydt, dRdt, dVdt, dDdt]


def seasyrvd_odes_noise(t, x, b_a, b_sy, g_a, g_sy, e_a, e_sy, a, v, d, std_f=0):    
    S = x[0]  # Susceptible
    E = x[1]  # Exposed
    A = x[2]  # Infected Asymptomatic
    Sy = x[3]  # Infected Symptomatic
    R = x[4]  # Recovered
    V = x[5]  # Vaccinated
    D = x[6]  # Deceased

    N = np.sum([S, E, A, Sy, R, V])  # Dynamic population (excludes deceased people)
    
    b_a = abs(np.random.normal(b_a, std_f*b_a))
    b_sy = abs(np.random.normal(b_sy, std_f*b_sy))
    g_a = abs(np.random.normal(g_a, std_f*g_a))
    g_sy = abs(np.random.normal(g_sy, std_f*g_sy))
    e_a = abs(np.random.normal(e_a, std_f*e_a))
    e_sy = abs(np.random.normal(e_sy, std_f*e_sy))
    a = abs(np.random.normal(a, std_f*a))
    v = abs(np.random.normal(v, std_f*v))
    d = abs(np.random.normal(d, std_f*d))

    dSdt = -(b_a/N)*A*S -(b_sy/N)*Sy*S - v*S
    dEdt = (b_a/N)*A*S + (b_sy/N)*Sy*S - e_a*E - e_sy*E
    dAdt = e_a*E - g_a*A - a*A
    dSydt = e_sy*E + a*A - g_sy*Sy - d*Sy
    dRdt = g_a*A + g_sy*Sy
    dVdt = v*S
    dDdt = d*Sy

    return [dSdt, dEdt, dAdt, dSydt, dRdt, dVdt, dDdt]


models = {
    'seasyrvd':seasyrvd_odes,
    'seasyrvd_noise':seasyrvd_odes_noise
}
