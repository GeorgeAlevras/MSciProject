import numpy as np
from generate_data import generate_data
from run_model import run_model
import argparse
import matplotlib.pyplot as plt
import time

def chi_sq(expected, observed):
    return np.sum([((e-o)**2)/(2*np.std(expected)**2) for e, o in zip(expected, observed)])


def use_args(args):
    R_0 = 3.5
    vaccination_rate = 0.006
    vac_ef = 0.8
    std_f = 0.2

    t_span = np.array([0, args.days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

    model_params = (R_0*(1/6), (1/6), vaccination_rate, std_f)
    model_state_params = [1000e3, 1e3, 0, 0, args.days]
    model_params_no_noise = (R_0*(1/6), (1/6), vaccination_rate, 0)
    generated_mode_no_noise, N = generate_data(model_params_no_noise, model_state_params)
    generated_model, N = generate_data(model_params, model_state_params)
    generated_model_infected_symptomatic = generated_model[1]
    generated_model_infected_symptomatic = np.random.normal(generated_model_infected_symptomatic, 0.05*generated_model_infected_symptomatic)
    data_std = np.std(generated_model_infected_symptomatic)
    
    init_params = model_params[:-1]
    model_params_old_1 = abs(np.random.normal(init_params, 0.5*np.array(init_params)))
    to_print_1 = model_params_old_1.copy()
    model_results_old_1 = np.array(run_model('sirv', model_params_old_1, model_state_params)[0])
    infected_old_1 = model_results_old_1[1]
    chi_old_1 = chi_sq(generated_model_infected_symptomatic, infected_old_1)
    accepted_params_1 = [[model_params_old_1.tolist(), chi_old_1]]

    model_params_old_2 = abs(np.random.normal(init_params, 0.5*np.array(init_params)))
    to_print_2 = model_params_old_2.copy()
    model_results_old_2 = np.array(run_model('sirv', model_params_old_2, model_state_params)[0])
    infected_old_2 = model_results_old_2[1]
    chi_old_2 = chi_sq(generated_model_infected_symptomatic, infected_old_2)
    accepted_params_2 = [[model_params_old_2.tolist(), chi_old_2]]

    model_params_old_3 = abs(np.random.normal(init_params, 0.5*np.array(init_params)))
    to_print_3 = model_params_old_3.copy()
    model_results_old_3 = np.array(run_model('sirv', model_params_old_3, model_state_params)[0])
    infected_old_3 = model_results_old_3[1]
    chi_old_3 = chi_sq(generated_model_infected_symptomatic, infected_old_3)
    accepted_params_3 = [[model_params_old_3.tolist(), chi_old_3]]

    model_params_old_4 = abs(np.random.normal(init_params, 0.5*np.array(init_params)))
    to_print_4 = model_params_old_4.copy()
    model_results_old_4 = np.array(run_model('sirv', model_params_old_4, model_state_params)[0])
    infected_old_4 = model_results_old_4[1]
    chi_old_4 = chi_sq(generated_model_infected_symptomatic, infected_old_4)
    accepted_params_4 = [[model_params_old_4.tolist(), chi_old_4]]

    boltzmann_c = 10000
    burn_in_time = 2000
    f = 1
    for i in range(20000):
        counter = min(len(accepted_params_1), len(accepted_params_2), len(accepted_params_3), len(accepted_params_4))

        model_params_new_1 = abs(np.random.normal(model_params_old_1, f*args.std*np.array(model_params_old_1)))
        model_results_new_1 = np.array(run_model('sirv', model_params_new_1, model_state_params)[0])
        infected_new_1 = model_results_new_1[1]
        chi_new_1 = chi_sq(generated_model_infected_symptomatic, infected_new_1)
        if (chi_new_1 < chi_old_1):
            accepted_params_1.append([model_params_new_1.tolist(), chi_new_1])
            model_params_old_1 = model_params_new_1
            chi_old_1 = chi_new_1
        else:
            if np.random.binomial(1, np.exp(-boltzmann_c*(chi_new_1-chi_old_1)/data_std)) == 1:
                accepted_params_1.append([model_params_new_1.tolist(), chi_new_1])
                model_params_old_1 = model_params_new_1
                chi_old_1 = chi_new_1
            else:
                pass
        
        model_params_new_2 = abs(np.random.normal(model_params_old_2, f*args.std*np.array(model_params_old_2)))
        model_results_new_2 = np.array(run_model('sirv', model_params_new_2, model_state_params)[0])
        infected_new_2 = model_results_new_2[1]
        chi_new_2 = chi_sq(generated_model_infected_symptomatic, infected_new_2)
        if (chi_new_2 < chi_old_2):
            accepted_params_2.append([model_params_new_2.tolist(), chi_new_2])
            model_params_old_2 = model_params_new_2
            chi_old_2 = chi_new_2
        else:
            if np.random.binomial(1, np.exp(-boltzmann_c*(chi_new_2-chi_old_2)/data_std)) == 1:
                accepted_params_2.append([model_params_new_2.tolist(), chi_new_2])
                model_params_old_2 = model_params_new_2
                chi_old_2 = chi_new_2
            else:
                pass

        model_params_new_3 = abs(np.random.normal(model_params_old_3, f*args.std*np.array(model_params_old_3)))
        model_results_new_3 = np.array(run_model('sirv', model_params_new_3, model_state_params)[0])
        infected_new_3 = model_results_new_3[1]
        chi_new_3 = chi_sq(generated_model_infected_symptomatic, infected_new_3)
        if chi_new_3 < chi_old_3:
            accepted_params_3.append([model_params_new_3.tolist(), chi_new_3])
            model_params_old_3 = model_params_new_3
            chi_old_3 = chi_new_3
        else:
            if np.random.binomial(1, np.exp(-boltzmann_c*(chi_new_3-chi_old_3)/data_std)) == 1:
                accepted_params_3.append([model_params_new_3.tolist(), chi_new_3])
                model_params_old_3 = model_params_new_3
                chi_old_3 = chi_new_3
            else:
                pass

        model_params_new_4 = abs(np.random.normal(model_params_old_4, f*args.std*np.array(model_params_old_4)))
        model_results_new_4 = np.array(run_model('sirv', model_params_new_4, model_state_params)[0])
        infected_new_4 = model_results_new_4[1]
        chi_new_4 = chi_sq(generated_model_infected_symptomatic, infected_new_4)
        if chi_new_4 < chi_old_4:
            accepted_params_4.append([model_params_new_4.tolist(), chi_new_4])
            model_params_old_4 = model_params_new_4
            chi_old_4 = chi_new_4
        else:
            if np.random.binomial(1, np.exp(-boltzmann_c*(chi_new_4-chi_old_4)/data_std)) == 1:
                accepted_params_4.append([model_params_new_4.tolist(), chi_new_4])
                model_params_old_4 = model_params_new_4
                chi_old_4 = chi_new_4
            else:
                pass
        if counter > burn_in_time:
            f *= 0.2
            boltzmann_c *= 0.05
        print('Done: ', i, '/20000')
    print('No. of steps accepted: ', len(accepted_params_1), len(accepted_params_2), len(accepted_params_3), len(accepted_params_4))
    print(round(accepted_params_1[-1][1]/accepted_params_1[0][1], 6), round(accepted_params_2[-1][1]/accepted_params_2[0][1], 6), round(accepted_params_3[-1][1]/accepted_params_3[0][1], 6), round(accepted_params_4[-1][1]/accepted_params_4[0][1], 6))
    print("Initial parameters: ", to_print_1, to_print_2, to_print_3, to_print_4)
    print("Final parameters: ", accepted_params_1[-1][0], accepted_params_2[-1][0], accepted_params_3[-1][0], accepted_params_4[-1][0])
    end_params = np.mean([accepted_params_1[-1][0], accepted_params_2[-1][0], accepted_params_3[-1][0], accepted_params_4[-1][0]], axis=0)
    all_params = [accepted_params_1[-1][0], accepted_params_2[-1][0], accepted_params_3[-1][0], accepted_params_4[-1][0], end_params]
    print("Average parameters: ", end_params)
    model_results_end = np.array(run_model('sirv', end_params, model_state_params)[0])
    infected_end = model_results_end[1]
    chi_sq_values = [accepted_params_1[-1][1], accepted_params_1[-1][1], accepted_params_1[-1][1], accepted_params_1[-1][1], chi_sq(generated_model_infected_symptomatic, infected_end)]
    best_params = all_params[np.argmin(chi_sq_values)]
    print('Best parameters: ', best_params)

    with open('Files/chain_1.txt', 'w') as file:
        for i, step in enumerate(accepted_params_1[2000:]):
            file.write(str(step[1])+',')
            for j, param in enumerate(step[0]):
                file.write(str(param))
                if j != len(step[0])-1:
                    file.write(',')
            file.write('\n')
    with open('Files/chain_2.txt', 'w') as file:
        for i, step in enumerate(accepted_params_2[2000:]):
            file.write(str(step[1])+',')
            for j, param in enumerate(step[0]):
                file.write(str(param))
                if j != len(step[0])-1:
                    file.write(',')
            file.write('\n')
    with open('Files/chain_3.txt', 'w') as file:
        for i, step in enumerate(accepted_params_3[2000:]):
            file.write(str(step[1])+',')
            for j, param in enumerate(step[0]):
                file.write(str(param))
                if j != len(step[0])-1:
                    file.write(',')
            file.write('\n')
    with open('Files/chain_4.txt', 'w') as file:
        for i, step in enumerate(accepted_params_4[2000:]):
            file.write(str(step[1])+',')
            for j, param in enumerate(step[0]):
                file.write(str(param))
                if j != len(step[0])-1:
                    file.write(',')
            file.write('\n')
    
    plt.figure(1)
    X = ['R_0', 'b', 'g', 'v']
    X_axis = np.arange(len(X))
    model_params = list(model_params)
    model_params.insert(0, 3.5)
    r_0 = (best_params[0])/(best_params[1])
    mcmc_params = list(best_params)
    mcmc_params.insert(0, r_0)
    plt.bar(X_axis-0.2, np.array(model_params[:-1])/np.array(model_params[:-1]), 0.2, label='Generated Data')
    plt.bar(X_axis+0.2, np.array(mcmc_params)/np.array(model_params[:-1]), 0.2, label='MCMC Data')
    plt.xticks(X_axis, X)
    plt.xlabel("Parameters")
    plt.ylabel("Parameters, normalised by generated data parameters")
    plt.title("MCMC Parameters Comparison")
    plt.legend()
    plt.savefig('Images/parameter_relative_ratios.png')

    plt.figure(2)
    plt.plot(t, generated_model_infected_symptomatic, '.', label='Generated Data')
    plt.plot(t, generated_mode_no_noise[1], label='Underlying Gen Data Signal')
    model_results_best = np.array(run_model('sirv', best_params, model_state_params)[0])
    infected_best = model_results_best[1]
    plt.plot(t, infected_best, label='MCMC Result')
    plt.xlabel('time')
    plt.ylabel('Infected People')
    plt.legend()
    plt.savefig('Images/results.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: MCMC Algorithm 1',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-d', '--days', type=int, help='Number of days of model to run')
    parser.add_argument('-s', '--std', type=float, help='Standard deviation of search, as a fraction of value')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
