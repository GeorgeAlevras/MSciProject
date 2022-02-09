from __future__ import print_function
from types import new_class
import numpy as np
import sys, os
from getdist import plots, MCSamples, loadMCSamples
import getdist, IPython
import pylab as plt
import seaborn as sns
import os
from glob import glob
from getdist import plots
from run_model import run_model


if __name__ == '__main__':
    model_params = np.load('ExperimentData/model_params.npy', allow_pickle=True)
    hyperparams = np.load('ExperimentData/hyperparams.npy', allow_pickle=True)
    generated_data_no_noise = np.load('ExperimentData/generated_data_no_noise.npy', allow_pickle=True)
    generated_data_noise = np.load('ExperimentData/generated_data_noise.npy', allow_pickle=True)
    generated_data_stds = np.load('ExperimentData/generated_data_stds.npy', allow_pickle=True)
    generated_data_weights = np.load('ExperimentData/generated_data_weights.npy', allow_pickle=True)
    accepted_params = np.load('ExperimentData/accepted_params.npy', allow_pickle=True)
    first_params = np.load('ExperimentData/first_params.npy', allow_pickle=True)
    number_of_acc_steps = np.load('ExperimentData/number_of_acc_steps.npy', allow_pickle=True)
    
    sys.path.insert(0, os.path.realpath(os.path.join(os.getcwd(), '..')))
    plt.rcParams['text.usetex']=True

    sns.set()
    sns.set_style("white")
    sns.set_context("talk")
    palette = sns.color_palette()
    plt.rcParams.update({'lines.linewidth': 3})
    plt.rcParams.update({'font.size': 22})

    file_names = [y for x in os.walk('./Chains/') for y in glob(os.path.join(x[0], 'chain*'))]

    for idx, file in enumerate(file_names):
        inChain = np.loadtxt(file,delimiter=',')
        nsamps, npar = inChain.shape
        outChain = np.zeros((nsamps,npar+1))
        outChain[:,1:] = np.copy(inChain)
        outChain[:,0] = 1.
        np.savetxt('./ConvertedFiles/convert_{}.txt'.format(idx+1),outChain)

    samples = loadMCSamples('./ConvertedFiles/convert')  #, settings={'ignore_rows':1000}
    best_stats = [x for x in str(samples.getLikeStats()).split(' ') if '.' in x]
    best_params = [float(best_stats[3]), float(best_stats[8]), float(best_stats[13]), float(best_stats[18])]

    plt.figure(1)
    X = ['b', 'g', 'h', 'd']
    X_axis = np.arange(len(X))
    model_params = list(model_params)
    mcmc_params = list(best_params)
    plt.bar(X_axis-0.2, np.array(model_params[:-1])/np.array(model_params[:-1]), 0.2, label='Generated Data')
    plt.bar(X_axis+0.2, np.array(mcmc_params)/np.array(model_params[:-1]), 0.2, label='MCMC Data')
    plt.xticks(X_axis, X)
    plt.xlabel("Parameters")
    plt.ylabel("Parameters, normalised by generated data parameters")
    plt.title("MCMC Parameters Comparison")
    plt.legend()
    plt.savefig('Images/parameter_relative_ratios.png')

    plt.figure(2)
    new_states = []
    for i in range(len(hyperparams)):
        if i == 0 or i == 1 or i == 3:
            new_states.append(int(hyperparams[i]))
        else:
            new_states.append(hyperparams[i])
    t_span = np.array([0, new_states[-1]])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    plt.plot(t, generated_data_noise[1], '.', label='Generated Data')
    plt.plot(t, generated_data_no_noise[1], label='Underlying Gen Data Signal')
    model_results_best = np.array(run_model('sirhd', best_params, new_states))
    infected_best = model_results_best[1]
    plt.plot(t, infected_best, label='MCMC Result')
    plt.xlabel('time')
    plt.ylabel('Infected People')
    plt.legend()
    plt.savefig('Images/results.png')

    for i in range(4):
        g = plots.get_single_plotter(width_inch=6)
        g.plot_1d(samples, 'p'+str(i+1), marker=best_params[i])
        g.export('./Images/p'+str(i+1)+'_dist.png')
    
    g = plots.getSubplotPlotter(width_inch=8)
    g.settings.alpha_filled_add=0.4
    g.settings.axes_fontsize = 20
    g.settings.lab_fontsize = 22
    g.settings.legend_fontsize = 20
    g.triangle_plot([samples], ['p1', 'p2','p3', 'p4'], 
        filled_compare=True, 
        legend_labels=['Samples'], 
        legend_loc='upper right', 
        line_args=[{'ls':'-', 'color':'green'}], 
        contour_colors=['green'])
    g.export('./Images/Triangle.pdf')
    plt.show()
