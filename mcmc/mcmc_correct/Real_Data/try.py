from run_model import run_model
import numpy as np
import time as time
import matplotlib.pyplot as plt
from produce_chains import generated_data_sirhd


def real_data(population=56000000):
    infected_percentage = np.array([1, 1.02, 1.03, 1.05, 1.07, 1.1, 1.14, 1.19, \
        1.24, 1.3, 1.37, 1.44, 1.51, 1.59, 1.67, 1.75, 1.82, 1.88, 1.93, 1.98, 2.02, 2.06, 2.09, 2.12, 2.16, 2.2])
    
    return population*0.01*infected_percentage


if __name__ == '__main__':
    with open('model_params.txt', 'r') as file:
        lines = file.readlines()
    model_params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]

    with open('hyperparams.txt', 'r') as file:
        lines = file.readlines()
    hyperparams = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]
    hyperparams[-1] = int(hyperparams[-1])

    infected = run_model('sirhd', model_params[:-1], hyperparams)[1]
    infected_real_dt = real_data()
    generated_infected_stds = generated_data_sirhd(model_params, hyperparams)[2][1]

    x = np.linspace(1, len(infected), len(infected))
    plt.plot(x, infected, label='Modeled Data')
    plt.plot(x, infected_real_dt, label='Real Data')
    plt.errorbar(x, infected_real_dt, generated_infected_stds)
    plt.legend()
    plt.show()
