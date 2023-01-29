import numpy as np


def seirv_odes(t, x, b, g, e, v):    
    S = x[0]  # Susceptible
    E = x[1]  # Exposed
    I = x[2]  # Infected
    R = x[3]  # Recovered
    V = x[4]  # Vaccinated
    
    N = np.sum([S, E, I, R, V])  # Dynamic population (excludes deceased people)
    
    dSdt = -(b/N)*I*S - v*S
    dEdt = (b/N)*I*S - e*E
    dIdt = e*E - g*I
    dRdt = g*I
    dVdt = v*S

    return [dSdt, dEdt, dIdt, dRdt, dVdt]


def seirv_odes_noise(t, x, b, g, e, v, std_f=0):    
    S = x[0]  # Susceptible
    E = x[1]  # Exposed
    I = x[2]  # Infected
    R = x[3]  # Recovered
    V = x[4]  # Vaccinated
    
    N = np.sum([S, E, I, R, V])  # Dynamic population (excludes deceased people)
    
    b = abs(np.random.normal(b, std_f*b))
    g = abs(np.random.normal(g, std_f*g))
    e = abs(np.random.normal(e, std_f*e))
    v = abs(np.random.normal(v, std_f*v))

    dSdt = -(b/N)*I*S - v*S
    dEdt = (b/N)*I*S - e*E
    dIdt = e*E - g*I
    dRdt = g*I
    dVdt = v*S

    return [dSdt, dEdt, dIdt, dRdt, dVdt]


models = {
    'seirv':seirv_odes,
    'seirv_noise':seirv_odes_noise
}
