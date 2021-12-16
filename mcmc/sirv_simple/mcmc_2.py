import numpy as np
from generate_data import generate_data
from run_model import run_model
import argparse
import matplotlib.pyplot as plt


def chi_sq(expected, observed):
    return np.sum([((e-o)**2)/(2*np.std(expected)**2) for e, o in zip(expected, observed)])


def generated_data(model_params, state_params, added_noise_f=0.05):
    '''
        model_params = (b, gamma, vaccination_rate, std_f)
        state_params = (population, infected, recovered, vaccinated, days) 
    '''

    model_params_no_noise = list(model_params).copy()
    model_params_no_noise[-1] = 0
    generated_model_no_noise, N_no_noise = generate_data(model_params_no_noise, state_params)
    generated_model_noise = generate_data(model_params, state_params)[0]
    generated_model_infected_symptomatic = np.random.normal(generated_model_noise[1], added_noise_f*generated_model_noise[1])
    data_std = np.std(generated_model_infected_symptomatic)

    return generated_model_no_noise, generated_model_infected_symptomatic, data_std
    

def mcmc(model_params, state_params, gen_infected, gen_std, iterations, std_f, chains=4, burn_in=1000):
    initial_params = np.full((chains, len(model_params[:-1])), model_params[:-1])
    old_params = np.random.uniform(0.5*initial_params, 2*initial_params)
    first_params = old_params.copy()
    old_results = np.array([run_model('sirv', params, state_params)[0] for params in old_params])
    old_infected = old_results[:,1]
    old_chi = np.array([chi_sq(gen_infected, i) for i in old_infected])
    accepted_params = [[o_p, o_c] for o_p, o_c in zip(old_params, old_chi)]

    boltzmann_c = 100000
    f = 1
    for i in range(iterations):
        min_acc = min([len(a) for a in accepted_params])
        new_params = abs(np.random.normal(old_params, f*std_f*initial_params))
        new_results = np.array([run_model('sirv', params, state_params)[0] for params in new_params])
        new_infected = new_results[:,1]
        new_chi = np.array([chi_sq(gen_infected, i) for i in new_infected])
        for chain in range(chains):
            if new_chi[chain] < old_chi[chain]:
                accepted_params[chain].append([new_params[chain], new_chi[chain]])
                old_params[chain] = new_params[chain]
                old_chi[chain] = new_chi[chain]
            else:
                if np.random.binomial(1, np.exp(-boltzmann_c*(new_chi[chain]-old_chi[chain])/gen_std)) == 1:
                    accepted_params[chain].append([new_params[chain], new_chi[chain]])
                    old_params[chain] = new_params[chain]
                    old_chi[chain] = new_chi[chain]
                else:
                    pass
        if min_acc > burn_in:
            f *= 0.2
            boltzmann_c *= 0.05
        if min_acc > 3*burn_in:
            f *= 0.2
        print('Done: ', i, '/'+str(iterations))
    
    for chain in range(chains):
        with open('Files/chain_'+str(chain+1)+'.txt', 'w') as file:
            for i, step in enumerate(accepted_params[chain][burn_in:]):
                file.write(str(step[1])+',')
                for j, param in enumerate(step[0]):
                    file.write(str(param))
                    if j != len(step[0])-1:
                        file.write(',')
                file.write('\n')

    return accepted_params, first_params


def use_args(args):
    if args.days == None:
        args.days = 50
    
    model_params = (0.58333, 0.16667, 0.006, 0.2)
    state_params = (1000e3, 1e3, 0, 0, args.days) 
    generated_model_no_noise, generated_model_infected_symptomatic, gen_std = generated_data(model_params, state_params)
    accepted, first = mcmc(model_params, state_params, generated_model_infected_symptomatic, gen_std, iterations=10000, std_f=0.3, chains=4, burn_in=1000)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: MCMC Algorithm 1',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-d', '--days', type=int, help='Number of days of model to run')
    parser.add_argument('-s', '--std', type=float, help='Standard deviation of search, as a fraction of value')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
