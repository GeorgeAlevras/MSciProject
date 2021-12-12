import numpy as np
from generate_data import generate_data
from run_model import run_model
import argparse
import matplotlib.pyplot as plt

def chi_sq(expected, observed):
    return np.sum([(e-o)**2 for e, o in zip(expected, observed)])


def use_args(args):
    R_0 = 3.5
    vaccination_rate = 0.006
    vac_ef = 0.8
    std_f = 0.2

    t_span = np.array([0, args.days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    model_params = (R_0*(1/6), R_0*(1/6), vaccination_rate, 0.083333333, 0.15, 0.1, (1/6), (1/6), 0.075, 0.003428571, 0.025, (1-vac_ef)*R_0*(1/6), (1-vac_ef)*R_0*(1/6), 0.1, 0.00035, 0.1, 0.00125, 0.2, (1/6), std_f)
    model_state_params = [1000e3, 1e3, 0, 0, args.days]
    generated_model, N = generate_data(model_params, model_state_params)
    generated_model_infected_symptomatic = generated_model[3] + generated_model[9]
    generated_model_infected_symptomatic = np.random.normal(generated_model_infected_symptomatic, 0.05*generated_model_infected_symptomatic)
    
    init_params = model_params[:-1]
    model_params_old = abs(np.random.normal(init_params, 0.01*np.array(init_params)))
    model_results_old = np.array(run_model('sirv_c', model_params_old, model_state_params)[0])
    infected_old = model_results_old[3] + model_results_old[9]
    chi_old = chi_sq(generated_model_infected_symptomatic, infected_old)
    accepted_params = [(model_params_old, chi_old)]

    for i in range(2000):
        model_params_new = abs(np.random.normal(model_params_old, args.std*np.array(model_params_old)))
        model_results_new = np.array(run_model('sirv_c', model_params_new, model_state_params)[0])
        infected_new = model_results_new[3] + model_results_new[9]
        chi_new = chi_sq(generated_model_infected_symptomatic, infected_new)

        if chi_new < chi_old:
            accepted_params.append((model_params_new, chi_new))
            model_params_old = model_params_new
        else:
            pass

        print('Done: ', i, '/2000' )
    print('No. of steps accepted: ', len(accepted_params))
    print(round(accepted_params[-1][1]/accepted_params[0][1], 3))
    print('Final parameters:', accepted_params[-1][0])
    
    with open('Files/mcmc.txt', 'w') as file:
        for step in accepted_params:
            for param in step[0]:
                file.write(str(param)+',')
            file.write(str(step[1])+'\n')
    

    X = ['R_0', 'b_a', 'b_sy', 'v', 'e_a', 'e_sy', 'a', 'g_a', 'g_sy', 'g_h', 'h', 'd_h', 'b_a_v', 'b_sy_v', 'a_v', 'h_v', 'g_h_v', 'd_h_v', 'g_a_v', 'g_sy_v']
    X_axis = np.arange(len(X))
    model_params = list(model_params)
    model_params.insert(0, 3.5)
    r_0 = (accepted_params[-1][0][0]+accepted_params[-1][0][1])/(accepted_params[-1][0][6]+accepted_params[-1][0][7])
    mcmc_params = list(accepted_params[-1][0])
    mcmc_params.insert(0, r_0)
    plt.bar(X_axis-0.2, np.array(model_params[:-1])/np.array(model_params[:-1]), 0.2, label='Generated Data')
    plt.bar(X_axis+0.2, np.array(mcmc_params)/np.array(model_params[:-1]), 0.2, label='MCMC Data')
    plt.xticks(X_axis, X)
    plt.xlabel("Parameters")
    plt.ylabel("Parameters, normalised by generated data parameters")
    plt.title("MCMC Parameters Comparison")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: MCMC Algorithm 1',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-d', '--days', type=int, help='Number of days of model to run')
    parser.add_argument('-s', '--std', type=float, help='Standard deviation of search, as a fraction of value')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
