import numpy as np
from generate_data import generate_data_sirv
from run_model import run_model
import argparse
import os


def chi_sq(expected, observed):
    return np.sum([((e-o)**2)/(2*np.std(expected)**2) for e, o in zip(expected, observed)])


def generated_data_sirv(model_params, state_params, added_noise_f=0.05):
    '''
        model_params = (b, gamma, vaccination_rate, std_f)
        state_params = (population, infected, recovered, vaccinated, days) 
    '''

    model_params_no_noise = list(model_params).copy()
    model_params_no_noise[-1] = 0
    generated_model_no_noise, N_no_noise = generate_data_sirv(model_params_no_noise, state_params)
    generated_model_noise = generate_data_sirv(model_params, state_params)[0]
    generated_model_infected = np.random.normal(generated_model_noise[1], added_noise_f*generated_model_noise[1])
    generated_model_susceptible = np.random.normal(generated_model_noise[0], added_noise_f*generated_model_noise[0])
    generated_model_recovered = np.random.normal(generated_model_noise[2], added_noise_f*generated_model_noise[2])
    generated_model_vaccinated = np.random.normal(generated_model_noise[3], added_noise_f*generated_model_noise[3])
    data_std = np.std(generated_model_infected)

    return generated_model_no_noise, generated_model_infected, data_std, generated_model_susceptible, generated_model_recovered, generated_model_vaccinated
    

def mcmc_sirv(model_params, state_params, gen_infected, gen_susc, gen_rec, gen_vac, gen_std, iterations, std_f, chains=4, burn_in=1000, p=0.2):
    initial_params = np.full((chains, len(model_params[:-1])), model_params[:-1])
    old_params = np.random.uniform(0.5*initial_params, 2*initial_params)  # initial set of parameters follows uniform dist
    first_params = old_params.copy()
    old_results = np.array([run_model('sirv', params, state_params)[0] for params in old_params])
    old_infected = old_results[:,1]
    old_susceptible = old_results[:,0]
    old_recovered = old_results[:,2]
    old_vaccinated = old_results[:,3]
    old_chi = np.array([chi_sq(gen_infected, i) + chi_sq(gen_susc, j) + chi_sq(gen_rec, k) + \
        chi_sq(gen_vac, z) for i, j, k, z in zip(old_infected, old_susceptible, old_recovered, old_vaccinated)])
    accepted_params = [[[o_p, o_c]] for o_p, o_c in zip(old_params, old_chi)]

    for i in range(iterations):
        min_acc = min([len(a) for a in accepted_params])
        new_params = abs(np.random.normal(old_params, std_f*initial_params))  # next guess is Gaussian centred at old_params
        new_results = np.array([run_model('sirv', params, state_params)[0] for params in new_params])
        new_infected = new_results[:,1]
        new_susceptible = new_results[:,0]
        new_recovered = new_results[:,2]
        new_vaccinated = new_results[:,3]
        new_chi = np.array([chi_sq(gen_infected, i) + chi_sq(gen_susc, j) + chi_sq(gen_rec, k) + \
            chi_sq(gen_vac, z) for i, j, k, z in zip(new_infected, new_susceptible, new_recovered, new_vaccinated)])
        
        for chain in range(chains):
            boltzmann_c = (-1)*gen_std*np.log(p)/(new_chi[chain]-old_chi[chain])
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
        if min_acc%burn_in == 0:  # every integer multiple of burn-in time
            std_f *= 1
            p *= 1
        print('Done: ', i, '/'+str(iterations))
    
    for chain in range(chains):
        with open('Chains/chain_'+str(chain+1)+'.txt', 'w') as file:
            for i, step in enumerate(accepted_params[chain][burn_in:]):
                file.write(str(step[1])+',')
                for j, param in enumerate(step[0]):
                    file.write(str(param))
                    if j != len(step[0])-1:
                        file.write(',')
                file.write('\n')

    return accepted_params, first_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: MCMC Algorithm 3',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-d', '--days', type=int, help='Number of days of model to run')
    parser.add_argument('-s', '--std', type=float, help='Standard deviation of search, as a fraction of value')
    parser.add_argument('-c', '--chains', type=int, help='Number of chains for search')
    parser.add_argument('-b', '--burn_in', type=int, help='Number of first X points to disregard, burn-in time')
    parser.add_argument('-i', '--iterations', type=int, help='Number of iterations for search')
    args = parser.parse_args()  # Parses all arguments provided at script on command-line
    
    if args.days == None:
        args.days = 50
    if args.std == None:
        args.std = 0.5
    if args.chains == None:
        args.chains = 8
    if args.iterations == None:
        args.iterations = 10000
    if args.burn_in == None:
        if args.iterations < 2000:
            args.burn_in = int(0.1*args.iterations)
        else:
            args.burn_in = 1000
    
    with open('model_params.txt', 'r') as file:
        lines = file.readlines()
    model_params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]

    with open('state_params.txt', 'r') as file:
        lines = file.readlines()
    state_params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]
    state_params[-1] = int(state_params[-1])

    generated_model_no_noise, generated_model_infected, gen_std, generated_model_susceptible, \
        generated_model_recovered, generated_model_vaccinated = generated_data_sirv(model_params, state_params)
    accepted, first = mcmc_sirv(model_params, state_params, generated_model_infected, \
        generated_model_susceptible, generated_model_recovered, generated_model_vaccinated, \
            gen_std, iterations=args.iterations, std_f=args.std, chains=args.chains, burn_in=args.burn_in)
    number_of_acc_steps = [len(accepted[i]) for i in range(len(accepted))]

    np.save(os.path.join('ExperimentData', 'generated_model_no_noise'), np.array(generated_model_no_noise))
    np.save(os.path.join('ExperimentData', 'generated_model_infected'), np.array(generated_model_infected))
    np.save(os.path.join('ExperimentData', 'gen_std'), np.array(gen_std))
    np.save(os.path.join('ExperimentData', 'generated_model_susceptible'), np.array(generated_model_susceptible))
    np.save(os.path.join('ExperimentData', 'generated_model_recovered'), np.array(generated_model_recovered))
    np.save(os.path.join('ExperimentData', 'generated_model_vaccinated'), np.array(generated_model_vaccinated))
    np.save(os.path.join('ExperimentData', 'accepted'), np.array(accepted))
    np.save(os.path.join('ExperimentData', 'first'), np.array(first))
    np.save(os.path.join('ExperimentData', 'model_params'), np.array(model_params))
    np.save(os.path.join('ExperimentData', 'state_params'), np.array(state_params))
    np.save(os.path.join('ExperimentData', 'number_of_acc_steps'), np.array(number_of_acc_steps))
