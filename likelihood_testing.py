from run_model import run_model
from run_noisy_model import run_noisy_model
import matplotlib.pyplot as plt
import argparse
import numpy as np
import scipy.stats as st

"""
    This file is used to test the likelihood function.
"""


def use_args(args):
    t_span = np.array([0, args.days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    
    generated_model = np.array(run_noisy_model('sirv_c', population=1000e3, R_0=3.5, vac_frac=0, nat_imm_rate=0, vac_ef=0.8, vaccination_rate=0.006, days=args.days, std_f=args.noise)[0])
    S, E, A, Sy, H, R, D, V, A_v, Sy_v, H_v = generated_model
    N = S + E + A + Sy + H + R + V + A_v + Sy_v + H_v
    generated_model_infected_symptomatic = generated_model[3] + generated_model[9]
    plt.plot(t, generated_model_infected_symptomatic/N, '.', label='Generated Data', color='black')
    
    solutions = []
    arguments = []
    for i in range(1000):
        model_results = run_model('sirv_c', population=1000e3, R_0=3.5, vac_frac=0, nat_imm_rate=0, vac_ef=0.8, vaccination_rate=0.006, days=args.days, stochastic=True)
        solutions.append(np.array(model_results[0]))
        arguments.append(model_results[2])
        S, E, A, Sy, H, R, D, V, A_v, Sy_v, H_v = solutions[i]
        N = S + E + A + Sy + H + R + V + A_v + Sy_v + H_v
        plt.plot(t, (Sy+Sy_v)/N, alpha=0.1, color='green')
        print('Done:', i+1,'/1000')
    
    solutions = np.array(solutions)
    infected_curves = solutions[:,3]+solutions[:,9]  # symptomatic: non-vaccinated[,3] and vaccinated[,9]
    sums_of_square_of_residuals = [np.sum((generated_model_infected_symptomatic-m)**2) for m in solutions]
    print('Iteration No.:', np.argmin(sums_of_square_of_residuals))
    print('Arguments:', arguments[np.argmin(sums_of_square_of_residuals)])
    plt.title('Symptomatic Infections Curves')
    plt.xlabel('Time [days]')
    plt.ylabel('Proportion of Population')
    plt.legend()
    plt.savefig('Plots/likelihood_'+str(args.noise)+'.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: Confidence Interval Experiment',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-d', '--days', type=int, help='Number of days of model to run')
    parser.add_argument('-n', '--noise', type=float, help='Noise, as a fraction of value')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
