from run_model import run_model
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy.stats as st

"""
    This file is used to plot compartmental models with stochastic parameters and see dynamics / time-evolutions of different compartments.
"""


def use_args(args):
    t_span = np.array([0, args.days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    solutions = []
    for i in range(10):
        solutions.append(run_model('sirv_c', population=1000e3, R_0=3.5, vac_frac=0, nat_imm_rate=0, vac_ef=0.8, vaccination_rate=0.006, days=args.days, stochastic=True)[0])
        S, E, A, Sy, H, R, D, V, A_v, Sy_v, H_v = solutions[i]
        N = S + E + A + Sy + H + R + V + A_v + Sy_v + H_v
        plt.plot(t, S/N, label='Susceptible', color='blue')
        plt.plot(t, E/N, label='Exposed', color='yellow')
        plt.plot(t, (A+A_v)/N, label='Asymptomatic', color='orange')
        plt.plot(t, (Sy+Sy_v)/N, alpha=0.3)
        plt.plot(t, V/N, label='Vaccinated', color='green')
        plt.plot(t, (H+H_v)/N, label='Hospitalised', color='brown')
        plt.plot(t, R/N, label='Recovered', color='pink')
        plt.plot(t, D/N, label='Deceased', color='black')
        plt.title(r'$SIRV \: Complex \: Model \:- \: Stochastic \: R_0 \: Time \: Evolution$')
        plt.xlabel('Time [days]')
        plt.ylabel('Proportion of Population')
        print('Done:', i+1,'/1000')
        if i == 0:
            plt.legend()
    plt.savefig('Plots/sirv_c_stochastic.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Plotter of Compartmental Models',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-d', '--days', type=int, help='Number of days of model to run')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
