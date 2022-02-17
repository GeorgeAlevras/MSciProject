import numpy as np


def sirhd_odes(t, x, b_1, b_2, g, h, d):
    # Compartments
    S = x[0]  # Susceptible
    I = x[1]  # Infected
    R = x[2]  # Recovered
    H = x[3]  # Hospitalised
    D = x[4]  # Deceased

    N = np.sum([S, I, R, H])  # Dynamic population (excludes deceased people)

    b_t = 3*[0.15] + 7*[0.165] + 10*[0.172] + 5*[0.175] + 5*[0.17] + 8*[0.165] + 12*[0.13] + 5*[0.145] + 5*[0.17] + 20*[0.18] + [0.19, 0.19, 0.2, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26] + \
        [0.27, 0.27, 0.28, 0.28, 0.28, 0.28, 0.29, 0.3, 0.31, 0.32, 0.32, 0.33, 0.31, 0.31, 0.29, 0.28, 0.27, 0.27]

    b = b_t[int(t)]
    dSdt = -(b/N)*I*S
    dIdt = (b/N)*I*S - h*I - g*I
    dRdt = g*I
    dHdt = h*I - d*H
    dDdt = d*H

    return [dSdt, dIdt, dRdt, dHdt, dDdt]


def sirhd_odes_noise(t, x, b_1, b_2, g, h, d, stochastic_noise_f=0):
    # Compartments
    S = x[0]  # Susceptible
    I = x[1]  # Infected
    R = x[2]  # Recovered
    H = x[3]  # Hospitalised
    D = x[4]  # Deceased

    N = np.sum([S, I, R, H])  # Dynamic population (excludes deceased people)

    # Introducing stochastic noise to the system of DEs by adding random noise to parameters in each time-step
    g = abs(np.random.normal(g, stochastic_noise_f*g))
    h = abs(np.random.normal(h, stochastic_noise_f*h))
    d = abs(np.random.normal(d, stochastic_noise_f*d))

    b_t = 3*[0.15] + 7*[0.165] + 10*[0.172] + 5*[0.175] + 5*[0.17] + 8*[0.165] + 12*[0.13] + 5*[0.145] + 5*[0.17] + 20*[0.18] + [0.19, 0.19, 0.2, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26] + \
        [0.27, 0.27, 0.28, 0.28, 0.28, 0.28, 0.29, 0.3, 0.31, 0.32, 0.32, 0.33, 0.31, 0.31, 0.29, 0.28, 0.27, 0.27]

    b = b_t[int(t)]
    dSdt = -(b/N)*I*S
    dIdt = (b/N)*I*S - h*I - g*I
    dRdt = g*I
    dHdt = h*I - d*H
    dDdt = d*H

    return [dSdt, dIdt, dRdt, dHdt, dDdt]


models = {
    'sirhd':sirhd_odes,
    'sirhd_noise':sirhd_odes_noise
}
