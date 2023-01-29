import numpy as np


def seirvd_odes(t, x, b, g, e, v, d):    
    S = x[0]  # Susceptible
    E = x[1]  # Exposed
    I = x[2]  # Infected
    R = x[3]  # Recovered
    V = x[4]  # Vaccinated
    D = x[5]  # Deceased
    
    N = np.sum([S, E, I, R, V])  # Dynamic population (excludes deceased people)
    
    dSdt = -(b/N)*I*S - v*S
    dEdt = (b/N)*I*S - e*E
    dIdt = e*E - g*I - d*I
    dRdt = g*I
    dVdt = v*S
    dDdt = d*I

    return [dSdt, dEdt, dIdt, dRdt, dVdt, dDdt]


def seirvd_odes_noise(t, x, b, g, e, v, d, std_f=0):    
    S = x[0]  # Susceptible
    E = x[1]  # Exposed
    I = x[2]  # Infected
    R = x[3]  # Recovered
    V = x[4]  # Vaccinated
    D = x[5]  # Deceased

    N = np.sum([S, E, I, R, V])  # Dynamic population (excludes deceased people)
    
    b = abs(np.random.normal(b, std_f*b))
    g = abs(np.random.normal(g, std_f*g))
    e = abs(np.random.normal(e, std_f*e))
    v = abs(np.random.normal(v, std_f*v))
    d = abs(np.random.normal(d, std_f*d))

    dSdt = -(b/N)*I*S - v*S
    dEdt = (b/N)*I*S - e*E
    dIdt = e*E - g*I - d*I
    dRdt = g*I
    dVdt = v*S
    dDdt = d*I

    return [dSdt, dEdt, dIdt, dRdt, dVdt, dDdt]


models = {
    'seirvd':seirvd_odes,
    'seirvd_noise':seirvd_odes_noise
}
