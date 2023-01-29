from run_model import run_model
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy.stats as st

"""
    This file is used to obtain confidence intervals from a Monte Carlo experiment of the model.
"""


def use_args(args):
    t_span = np.array([0, args.days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    solutions = []
    for i in range(1000):
        solutions.append(run_model('sirv_c', population=1000e3, R_0=3.5, vac_frac=0, nat_imm_rate=0, vac_ef=0.8, vaccination_rate=0.006, days=args.days, stochastic=True)[0])
        S, E, A, Sy, H, R, D, V, A_v, Sy_v, H_v = solutions[i]
        N = S + E + A + Sy + H + R + V + A_v + Sy_v + H_v
        plt.plot(t, (Sy+Sy_v)/N, alpha=0.3, color='green')
        print('Done:', i+1,'/1000')
    solutions = np.array(solutions)
    infected_curves = solutions[:,3]+solutions[:,9]  # symptomatic: non-vaccinated[,3] and vaccinated[,9]
    infected_curves_time_series = [infected_curves[:,i] for i in range(len(infected_curves[0]))]
    fiftieth_percentile = [np.percentile(el, 50) for el in infected_curves_time_series]
    confidence_curve_low = [np.percentile(el, 2.5) for el in infected_curves_time_series]
    confidence_curve_high = [np.percentile(el, 97.5) for el in infected_curves_time_series]
    plt.title(r'$Stochastic \: Parameters \: Confidence \: Interval$')
    plt.xlabel('Time [days]')
    plt.ylabel('Proportion of Population')
    plt.plot(t, fiftieth_percentile/N, '--', label='mean', linewidth=2, color='black')
    plt.plot(t, confidence_curve_low/N, '--', label='a=2.5%', linewidth=2, color='red')
    plt.plot(t, confidence_curve_high/N, '--', label='a=97.5%', linewidth=2, color='blue')
    plt.legend()
    plt.savefig('Plots/sirv_c_confidence_intervals.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Confidence Interval Experiment',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-d', '--days', type=int, help='Number of days of model to run')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
