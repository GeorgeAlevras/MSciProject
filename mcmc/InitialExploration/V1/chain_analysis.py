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
    generated_model_no_noise = np.load('ExperimentData/generated_model_no_noise.npy', allow_pickle=True)
    generated_model_infected = np.load('ExperimentData/generated_model_infected.npy', allow_pickle=True)
    gen_std = np.load('ExperimentData/gen_std.npy', allow_pickle=True)
    generated_model_susceptible = np.load('ExperimentData/generated_model_susceptible.npy', allow_pickle=True)
    generated_model_recovered = np.load('ExperimentData/generated_model_recovered.npy', allow_pickle=True)
    generated_model_vaccinated = np.load('ExperimentData/generated_model_vaccinated.npy', allow_pickle=True)
    accepted = np.load('ExperimentData/accepted.npy', allow_pickle=True)
    first = np.load('ExperimentData/first.npy', allow_pickle=True)
    model_params = np.load('ExperimentData/model_params.npy', allow_pickle=True)
    state_params = np.load('ExperimentData/state_params.npy', allow_pickle=True)
    number_of_acc_steps = np.load('ExperimentData/number_of_acc_steps.npy', allow_pickle=True)
    print(number_of_acc_steps)
    sys.path.insert(0, os.path.realpath(os.path.join(os.getcwd(), '..')))
    plt.rcParams['text.usetex']=True

    sns.set()
    sns.set_style("white")
    sns.set_context("talk")
    #sns.set_context("poster")
    palette = sns.color_palette()
    #plt.rcParams.update({'font.size': 22})
    plt.rcParams.update({'lines.linewidth': 3})
    #plt.rcParams.update({'usetex': True})
    #cp = sns.color_palette()
    plt.rcParams.update({'font.size': 22})

    file_names = [y for x in os.walk('./Chains/') for y in glob(os.path.join(x[0], 'chain*'))]

    #Quick convert to add an importance weight column left of -Log(like) column
    for idx, file in enumerate(file_names):
        inChain = np.loadtxt(file,delimiter=',')
        nsamps, npar = inChain.shape
        outChain = np.zeros((nsamps,npar+1))
        outChain[:,1:] = np.copy(inChain)
        outChain[:,0] = 1.
        np.savetxt('./ConvertedFiles/convert_{}.txt'.format(idx+1),outChain)

    samples = loadMCSamples('./ConvertedFiles/convert')  #, settings={'ignore_rows':1000}
    best_stats = [x for x in str(samples.getLikeStats()).split(' ') if '.' in x]
    best_params = [float(best_stats[3]), float(best_stats[8]), float(best_stats[13])]

    plt.figure(1)
    X = ['b', 'g', 'v']
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
    for i in range(len(state_params)):
        if i == 0 or i == 1 or i == 4:
            new_states.append(int(state_params[i]))
        else:
            new_states.append(state_params[i])
    t_span = np.array([0, new_states[-1]])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    plt.plot(t, generated_model_infected, '.', label='Generated Data')
    plt.plot(t, generated_model_no_noise[1], label='Underlying Gen Data Signal')
    model_results_best = np.array(run_model('sirv', best_params, new_states)[0])
    infected_best = model_results_best[1]
    plt.plot(t, infected_best, label='MCMC Result')
    plt.xlabel('time')
    plt.ylabel('Infected People')
    plt.legend()
    plt.savefig('Images/results.png')

    # 1D marginalized plot
    g = plots.get_single_plotter(width_inch=6)
    g.plot_1d(samples, 'p1', marker=best_params[0])
    g.export('./Images/p1_dist.png')
    g = plots.get_single_plotter(width_inch=6)
    g.plot_1d(samples, 'p2', marker=best_params[1])
    g.export('./Images/p2_dist.png')
    g = plots.get_single_plotter(width_inch=6)
    g.plot_1d(samples, 'p3', marker=best_params[2])
    g.export('./Images/p3_dist.png')
    print('\n')

    g = plots.getSinglePlotter()
    g.plot_2d(samples, ['p1', 'p2'], shaded=True)
    g.export('./Images/p1p2.png')
    g = plots.getSinglePlotter()
    g.plot_2d(samples, ['p1', 'p3'], shaded=True)
    g.export('./Images/p1p3.png')
    g = plots.getSinglePlotter()
    g.plot_2d(samples, ['p2', 'p3'], shaded=True)
    g.export('./Images/p2p3.png')


    g = plots.getSubplotPlotter(width_inch=8)
    #g.settings.figure_legend_frame = False
    g.settings.alpha_filled_add=0.4
    g.settings.axes_fontsize = 20
    g.settings.lab_fontsize = 22
    g.settings.legend_fontsize = 20
    g.triangle_plot([samples], ['p1', 'p2','p3'], 
        filled_compare=True, 
        legend_labels=['Samples'], 
        legend_loc='upper right', 
        line_args=[{'ls':'-', 'color':'green'}], 
        contour_colors=['green'])
    g.export('./Images/Triangle.pdf')

    # Many other things you can do besides plot, e.g. get latex
    # Default limits are 1: 68%, 2: 95%, 3: 99% probability enclosed
    # See  https://getdist.readthedocs.io/en/latest/analysis_settings.html
    # and examples for below for changing analysis settings 
    # (e.g. 2hidh limits, and how they are defined)
    print(samples.getInlineLatex('p1',limit=2))
    print(samples.getInlineLatex('p2',limit=2))
    print(samples.getInlineLatex('p3',limit=2))
    print(samples.getTable().tableTex())
    print(samples.PCA(['p1','p2','p3']))
    print('\n')

    stats = samples.getMargeStats()
    lims0 = stats.parWithName('p1').limits
    lims1 = stats.parWithName('p2').limits
    lims2 = stats.parWithName('p3').limits
    for conf, lim0, lim1, lim2 in zip(samples.contours, lims0, lims1, lims2):
        print('p1 %s%% lower: %.3f upper: %.3f (%s)'%(conf, lim0.lower, lim0.upper, lim0.limitType()))
        print('p2 %s%% lower: %.3f upper: %.3f (%s)'%(conf, lim1.lower, lim1.upper, lim1.limitType()))
        print('p3 %s%% lower: %.3f upper: %.3f (%s)'%(conf, lim2.lower, lim2.upper, lim2.limitType()))

    plt.show()
