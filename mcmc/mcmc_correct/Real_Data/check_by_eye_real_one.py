from run_model import run_model
import numpy as np
import time as time
import matplotlib.pyplot as plt
from produce_chains import generated_data_sirhd
import pandas as pd
import matplotlib


def real_data(population=56000000):
    infected_percentage = np.array([0.25, 0.26, 0.28, 0.29, 0.31, 0.33, 0.36, 0.38, 0.4, 0.43, 0.46, 0.49, \
        0.51, 0.54, 0.57, 0.59, 0.62, 0.64, 0.67, 0.69, 0.72, 0.73, 0.76, 0.79, 0.82, 0.91, 0.94, 0.97, 1.0, \
            1.03, 1.06, 1.08, 1.1, 1.1, 1.11, 1.12, 1.12, 1.13, 1.14, 1.15])
    
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
    date_space_1 = pd.date_range(start='20/09/2021', end='31/10/2021')
    date_space_2 = pd.date_range(start='20/09/2021', end='31/10/2021')
    
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
