from run_model import run_model
import numpy as np
import time as time
import matplotlib.pyplot as plt
from produce_chains import generated_data_sirhd


def real_data(population=56000000):
    infected_percentage = np.array([1.09, 1.1, 1.11, 1.12, 1.14, 1.16, 1.18, 1.2, 1.23, 1.26, 1.29, 1.32, 1.35, 1.38, 1.41, \
        1.44, 1.47, 1.5, 1.53, 1.56, 1.59, 1.61, 1.64, 1.67, 1.69, 1.72, 1.75, 1.78, 1.81, 1.85, 1.89, 1.93, 1.97, 2.02, 2.08, \
            2.07, 2.08, 2.08, 2.08, 2.06, 2.03, 2, 1.96, 1.92, 1.87, 1.83, 1.78, 1.74, 1.69, 1.66, 1.62, 1.6, 1.57, 1.56, \
                1.54, 1.54, 1.54, 1.54, 1.54, 1.55, 1.56, 1.57, 1.59, 1.6, 1.61, 1.62, 1.63, 1.64, 1.64, 1.64, 1.64, 1.64, \
                    1.64, 1.64, 1.63, 1.64, 1.64, 1.68, 1.7, 1.73, 1.74, 1.75, 1.77, 1.8, 1.85, 1.91, 2, 2.1, 2.23, 2.39, 2.58, \
                        2.79, 3.02, 3.28, 3.56, 3.85, 4.14, 4.43, 4.71, 4.98, 5.24, 5.49, 5.74, 6, 6.27, 6.56, 6.87])
    
    return population*0.01*infected_percentage


if __name__ == '__main__':
    with open('model_params.txt', 'r') as file:
        lines = file.readlines()
    model_params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]

    with open('hyperparams.txt', 'r') as file:
        lines = file.readlines()
    hyperparams = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]
    hyperparams[-1] = int(hyperparams[-1])
    hyperparams[-2] = int(hyperparams[-2])
    
    infected = run_model('sirhd', model_params[:-1], hyperparams)[1]
    infected_real_dt = real_data()
    generated_infected_stds = generated_data_sirhd(model_params, hyperparams)[2][1]
    x = np.linspace(1, len(infected), len(infected))
    plt.plot(x, infected, '--', label='Modeled Data', linewidth=3)
    plt.plot(x, infected_real_dt, label='Real Data')
    plt.errorbar(x, infected_real_dt, yerr=generated_infected_stds, color='black', linestyle = '', capsize = 4, alpha=1, label='Assumed Errors')
    plt.legend()
    plt.show()
