from run_model import run_model
import numpy as np
import time as time
import matplotlib.pyplot as plt
from produce_chains import generated_data_sirhd
import pandas as pd
import matplotlib


def real_data(population=56000000):
    infected_percentage = np.array([1.09, 1.1, 1.11, 1.12, 1.14, 1.16, 1.18, 1.2, 1.23, 1.26, 1.29, 1.32, 1.35, 1.38, 1.41, \
        1.44, 1.47, 1.5, 1.53, 1.56, 1.59, 1.61, 1.64, 1.67, 1.69, 1.72, 1.75, 1.78, 1.81, 1.85, 1.89, 1.93, 1.97, 2.02, 2.08, \
            2.07, 2.08, 2.08, 2.08, 2.06, 2.03, 2, 1.96, 1.92, 1.87, 1.83, 1.78, 1.74, 1.69, 1.66, 1.62, 1.6, 1.57, 1.56, \
                1.54, 1.54, 1.54, 1.54, 1.54, 1.55, 1.56, 1.57, 1.59, 1.6, 1.61, 1.62, 1.63, 1.64, 1.64, 1.64, 1.64, 1.64, \
                    1.64, 1.64, 1.63, 1.64, 1.64, 1.68, 1.7, 1.73, 1.74, 1.75, 1.77, 1.8, 1.85, 1.91, 2, 2.1, 2.23, 2.39, 2.58, \
                        2.79, 3.02, 3.28, 3.56, 3.85, 4.14, 4.43, 4.71, 4.98, 5.24, 5.49, 5.74, 6, 6.27, 6.56, 6.87, \
                            6.99, 7.06, 7.08, 7.06, 6.99, 6.89, 6.75, 6.59, 6.42, 6.23, 6.04, 5.85, 5.67, 5.49])
    
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
    
    S, I, R, H, D =  run_model('sirhd', model_params[:-1], hyperparams)
    N = S + I + R + H + D
    infected = run_model('sirhd', model_params[:-1], hyperparams)[1]
    
    S_min, I_min, R_min, H_min, D_min =  run_model('sirhd_min', model_params[:-1], hyperparams)
    N_min = S_min + I_min + R_min + H_min + D_min
    infected_min = run_model('sirhd_min', model_params[:-1], hyperparams)[1]
    
    S_max, I_max, R_max, H_max, D_max =  run_model('sirhd_max', model_params[:-1], hyperparams)
    N_max = S_max + I_max + R_max + H_max + D_max
    infected_max = run_model('sirhd_max', model_params[:-1], hyperparams)[1]
    
    infected_real_dt = real_data()
    hyperparams[-2] = 107
    generated_infected_stds = generated_data_sirhd(model_params, hyperparams)[2][1]
    date_space_1 = pd.date_range(start='16/09/2021', end='31/12/2021')
    date_space_2 = pd.date_range(start='16/09/2021', end='14/01/2022')
    
    params = {
    'axes.labelsize': 18,
    'font.size': 18,
    'font.family': 'serif',
    'legend.fontsize': 26,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
    'figure.figsize': [8, 5]
    } 

    fam  = {'fontname':'Times New Roman'}
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams.update(params)
    
    plt.plot(date_space_2, 100*infected_real_dt/N, color='black', label='Real Data', linewidth=2)
    plt.plot(date_space_2[:-13], 100*infected[:-13]/N[:-13], '--', color='limegreen', label='Modeled Data', linewidth=3)
    plt.plot(date_space_2[-14:], 100*infected[-14:]/N[-14:], '--', color='darkviolet', label='Predicted Modeled Data', linewidth=3)
    plt.fill_between(date_space_2[-14:], 100*infected_min[-14:]/N_min[-14:], 100*infected_max[-14:]/N_max[-14:], color = 'darkviolet', zorder = 2, alpha = 0.4)
    # plt.plot(date_space_2[-14:], 100*infected_min[-14:]/N_min[-14:], '--', color='darkviolet', linewidth=2)   
    # plt.plot(date_space_2[-14:], 100*infected_max[-14:]/N_max[-14:], '--', color='darkviolet', linewidth=2)      
    plt.plot([date_space_2[-14], date_space_2[-14]], 100*np.linspace(0, 1, 2), '--', label='Cut-off')
    plt.xlabel('Date', **fam, fontsize=22)
    plt.ylabel(str('%')+' Of People Infected', **fam, fontsize=22)
    plt.legend()
    plt.grid()
    plt.xlim(pd.date_range(start='15/09/2021', end='15/09/2021'), pd.date_range(start='15/01/2022', end='15/01/2022'))
    plt.ylim(0, 10)
    plt.savefig('poster.jpeg')
    plt.show()
