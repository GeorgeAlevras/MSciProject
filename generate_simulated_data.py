from run_model import run_model
import matplotlib.pyplot as plt
import argparse
import numpy as np


def use_args(args):
    t_span = np.array([0, args.days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    solutions = np.array(run_model('sirv_c', population=1000e3, R_0=3.5, vac_frac=0, nat_imm_rate=0, vac_ef=0.8, vaccination_rate=0.006, days=args.days, stochastic=False)[0])
    S, E, A, Sy, H, R, D, V, A_v, Sy_v, H_v = solutions
    N = S + E + A + Sy + H + R + V + A_v + Sy_v + H_v
    infected = solutions[3] + solutions[9]
    signal = infected
    noise = signal - np.random.normal(signal, 0.05*signal)
    simulated_data = signal + noise
    plt.plot(t, infected/N, label='Model Signal')
    plt.plot(t, simulated_data/N, '.', label='Simulated Data')
    plt.title('Simulated Data over Model Signal')
    plt.xlabel('Time [days]')
    plt.ylabel('Proportion of Population')
    plt.legend()
    plt.savefig('Plots/generated_simulated_data.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Generation of Model Data',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-d', '--days', type=int, help='Number of days of model to run')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
