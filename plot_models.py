from run_model import run_model
import matplotlib.pyplot as plt
import argparse
import numpy as np

"""
    This file is used to plot compartmental models and see dynamics / time-evolutions of different compartments.
"""


def use_args(args):
    t_span = np.array([0, args.days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    solutions = run_model('sirv_c', population=1000e3, R_0=3.5, vac_frac=0, nat_imm_rate=0, vac_ef=0.8, vaccination_rate=0.006, days=args.days)[0]
    S, E, A, Sy, H, R, D, V, A_v, Sy_v, H_v = solutions
    N = S + E + A + Sy + H + R + V + A_v + Sy_v + H_v
    print('Deceased:', round(D[-1], 0))  # Total number of deaths during 'pandemic'
    print('Hospitalised Peak:', round(max(H), 0))  # Highest number of hospitalised individuals in a given day
    plt.plot(t, S/N, label='Susceptible')
    plt.plot(t, E/N, label='Exposed')
    plt.plot(t, (A+A_v)/N, label='Asymptomatic')
    plt.plot(t, (Sy+Sy_v)/N, label='Symptomatic')
    plt.plot(t, V/N, label='Vaccinated')
    plt.plot(t, (H+H_v)/N, label='Hospitalised')
    plt.plot(t, R/N, label='Recovered')
    plt.plot(t, D/N, label='Deceased')
    plt.xlabel('Time [days]')
    plt.ylabel('Proportion of Population')
    plt.legend()
    plt.savefig('Plots/sirv_c.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Plotter of Compartmental Models',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-d', '--days', type=int, help='Number of days of model to run')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
