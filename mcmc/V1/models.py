import numpy as np


def sirv_odes(t, x, b, g, v):    
    S = x[0]  # Susceptible
    I = x[1]  # Infected
    R = x[2]  # Recovered
    V = x[3]  # Vaccinated
    
    N = np.sum([S, I, R, V])  # Dynamic population (excludes deceased people)
    
    dSdt = -(b/N)*I*S - v*S
    dIdt = (b/N)*I*S - g*I
    dRdt = g*I
    dVdt = v*S

    return [dSdt, dIdt, dRdt, dVdt]


def sirv_odes_noise(t, x, b, g, v, std_f=0):    
    S = x[0]  # Susceptible
    I = x[1]  # Infected
    R = x[2]  # Recovered
    V = x[3]  # Vaccinated
    
    N = np.sum([S, I, R, V])  # Dynamic population (excludes deceased people)
    
    b = abs(np.random.normal(b, std_f*b))
    g = abs(np.random.normal(g, std_f*g))
    v = abs(np.random.normal(v, std_f*v))

    dSdt = -(b/N)*I*S - v*S
    dIdt = (b/N)*I*S - g*I
    dRdt = g*I
    dVdt = v*S

    return [dSdt, dIdt, dRdt, dVdt]


models = {
    'sirv':sirv_odes,
    'sirv_noise':sirv_odes_noise
}
