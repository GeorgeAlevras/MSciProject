from run_model import run_model
from generate_data import generate_data
import numpy as np
import matplotlib.pyplot as plt

mu = 10
s = 2
def my_dist(x, mu, s):
    return (1/np.sqrt(2*np.pi*s))*np.exp(-((x-mu)**2)/(2*s**2))

def chi_sq(expected, observed):
    return np.sum([(e-o)**2 for e, o in zip(expected, observed)])

space = np.linspace(0, 40, 401)
original = my_dist(space, mu, s)

init_mu = 12
init_s = 3
my_values = my_dist(space, init_mu, init_s)
chi_init = chi_sq(original, my_values)
accepted_steps = []

for i in range(1000):
    new_param_mu = np.random.normal(init_mu, 0.1*init_mu)
    new_param_s = np.random.normal(init_s, 0.1*init_s)
    new_values = my_dist(space, new_param_mu, new_param_s)
    chi_new = chi_sq(original, new_values)

    if chi_new < chi_init:
        accepted_steps.append((new_param_mu, new_param_s))
        init_mu = new_param_mu
        init_s = new_param_s
        chi_init = chi_new
    else:
        pass

print(accepted_steps)
