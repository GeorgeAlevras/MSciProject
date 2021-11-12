from run_model import run_model
import argparse
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar

"""
    This file is an experiment to empirically verify the theoretical approximation of the minimum 
    fraction of of the population needed to be vaccinated for a given R_0 and vaccine efficacy.

    1. my_search: a binary search algorithm (using a convergence criterion) to find the minimum
        fraction of the population needed.
"""


def my_search(model, R_0, vac_ef, nat_imm_rate, rel_acc_wanted=0.01):
    days = 50
    vaccination_rate = 0
    population = 1000e3
    high = 998e3
    low = 1e3
    mid = 0.5*(low+high)

    vac_frac = high/population
    max_R_t = run_model(model, population, R_0, vac_frac, nat_imm_rate, vac_ef, vaccination_rate, days)[1]
    
    if max_R_t > 1:
        return None
    else:
        vac_frac = mid/population
        max_R_t = run_model(model, population, R_0, vac_frac, nat_imm_rate, vac_ef, vaccination_rate, days)[1]
        rel_acc = abs(max_R_t - 1)
        counter = 0
        while rel_acc > rel_acc_wanted:
            counter += 1
            if max_R_t > 1:
                low = mid
                mid = 0.5*(low+high)
                vac_frac = mid/population
                max_R_t = run_model(model, population, R_0, vac_frac, nat_imm_rate, vac_ef, vaccination_rate, days)[1]
                rel_acc = abs(max_R_t - 1)
                if counter > 20:
                    break
            else:
                high = mid
                mid = 0.5*(low+high)
                vac_frac = mid/population
                max_R_t = run_model(model, population, R_0, vac_frac, nat_imm_rate, vac_ef, vaccination_rate, days)[1]
                rel_acc = abs(max_R_t - 1)
                if counter > 20:
                    break
        return vac_frac


def theoretical(V, E, nat_imm=0):
    return 1/(1-nat_imm-(1-nat_imm)*E*V)


def use_args(args):
    n_i = 0
    vaccinated_fraction_space = np.linspace(0, 0.999, 101)
    R_0s = np.linspace(1.1, 8, 70)
    vaccine_efficacies = np.linspace(0.3, 1, 15)

    theoretical_curves = []
    for i in range(len(vaccine_efficacies)):
        theoretical_curves.append(theoretical(vaccinated_fraction_space, vaccine_efficacies[i], nat_imm=n_i))
    
    vaccine_efficacy_curves = []
    R_0_to_plot = []

    if not args.sirv_model:
        bar = Bar('Running Search Algorithm', max=536)
        for vac_ef in vaccine_efficacies:
            curve = []
            curve_R_0 = []
            for R_0 in R_0s:
                vac_frac = my_search(model='sirv_c', R_0=R_0, vac_ef=vac_ef, nat_imm_rate=n_i, rel_acc_wanted=0.001)
                if vac_frac == None:
                    break
                curve.append(vac_frac)
                curve_R_0.append(R_0)
                bar.next()
            vaccine_efficacy_curves.append(curve)
            R_0_to_plot.append(curve_R_0)
        bar.finish()
        
        residuals_metrics = []
        for i in range(len(vaccine_efficacies)):
            t = theoretical(np.array(vaccine_efficacy_curves[i]), vaccine_efficacies[i], nat_imm=n_i)
            fractional_residuals = (t-R_0_to_plot[i])/t
            residuals_metrics.append([np.average(fractional_residuals), max(abs(fractional_residuals))])
        residuals_metrics = np.array(residuals_metrics)
        avg_res = np.average(residuals_metrics[:,0])
        max_res = max(residuals_metrics[:,1])
        print('Average residual:', avg_res, '\nMaximum residual:', max_res)

        fig, ax = plt.subplots()
        colors = ['black', 'blue', 'green', 'red', 'orange', 'purple']
        for i in range(len(vaccine_efficacy_curves)):
            plt.plot(R_0_to_plot[i], vaccine_efficacy_curves[i], '.', label=r'$Model: \: {}$'.format(str(round(vaccine_efficacies[i]*100, 0)))+'%', color=colors[i%len(colors)])
            plt.plot(theoretical_curves[i], vaccinated_fraction_space, label=r'Theoretical: {}%'.format(str(round(vaccine_efficacies[i]*100, 0))), color=colors[i%len(colors)])
        plt.minorticks_off()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(0.75, max(R_0s)+0.25)
        plt.ylim(0, 1)
        plt.xlabel(r'$R_0$')
        plt.ylabel('Proportion of Population Vaccinated')
        plt.title('SIRV Complex Model, Nat. Immunity: ' + str(n_i*100) + '%')
        plt.legend()
        plt.savefig('Plots/experiment_sirv_complex_'+str(n_i)+'.png')
        plt.show()
    else:
        bar = Bar('Running Search Algorithm', max=536)
        for vac_ef in vaccine_efficacies:
            curve = []
            curve_R_0 = []
            for R_0 in R_0s:
                vac_frac = my_search(model='sirv', R_0=R_0, vac_ef=vac_ef, nat_imm_rate=n_i, rel_acc_wanted=0.001)
                if vac_frac == None:
                    break
                curve.append(vac_frac)
                curve_R_0.append(R_0)
                bar.next()
            vaccine_efficacy_curves.append(curve)
            R_0_to_plot.append(curve_R_0)
        bar.finish()
        
        residuals_metrics = []
        for i in range(len(vaccine_efficacies)):
            t = theoretical(np.array(vaccine_efficacy_curves[i]), vaccine_efficacies[i], nat_imm=n_i)
            fractional_residuals = (t-R_0_to_plot[i])/t
            residuals_metrics.append([np.average(fractional_residuals), max(abs(fractional_residuals))])
        residuals_metrics = np.array(residuals_metrics)
        avg_res = np.average(residuals_metrics[:,0])
        max_res = max(residuals_metrics[:,1])
        print('Average residual:', avg_res, '\nMaximum residual:', max_res)

        fig, ax = plt.subplots()
        colors = ['black', 'blue', 'green', 'red', 'orange', 'purple']
        for i in range(len(vaccine_efficacy_curves)):
            plt.plot(R_0_to_plot[i], vaccine_efficacy_curves[i], '.', label=r'$Model: \: {}$'.format(str(round(vaccine_efficacies[i]*100, 0)))+'%', color=colors[i%len(colors)])
            plt.plot(theoretical_curves[i], vaccinated_fraction_space, label=r'Theoretical: {}%'.format(str(round(vaccine_efficacies[i]*100, 0))), color=colors[i%len(colors)])
        plt.minorticks_off()
        ax.tick_params(direction='in')
        ax.tick_params(which='minor', direction='in')
        plt.xticks(fontsize=12, fontname='Times New Roman')
        plt.yticks(fontsize=12, fontname='Times New Roman')
        plt.xlim(0.75, max(R_0s)+0.25)
        plt.ylim(0, 1)
        plt.xlabel(r'$R_0$')
        plt.ylabel('Proportion of Population Vaccinated')
        plt.title('SIRV Model, Nat. Immunity: ' + str(n_i*100) + '%')
        plt.legend()
        plt.savefig('Plots/experiment_sirv_'+str(n_i)+'.png')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Vaccine Efficacy Experiment',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-sirv', '--sirv_model', action='store_true', help='If present, will run the basic SIRV model')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
