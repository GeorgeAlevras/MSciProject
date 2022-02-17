from typing import final
import numpy as np
from generate_data import generate_data_sirhd
from run_model import run_model
import argparse
import os
import matplotlib.pyplot as plt
import time as time


def chi_sq(data, model, std_s):
    # Reduced chi-squared of model given some data
    chi_sq = []
    for i in range(len(data)):
        if std_s[i] != 0:
            chi_sq.append((data[i] - model[i])**2 / std_s[i]**2)
    
    return (1/len(chi_sq))*np.sum(chi_sq)


def real_data(population=56000000):
    infected_percentage = np.array([1.09, 1.1, 1.11, 1.12, 1.14, 1.16, 1.18, 1.2, 1.23, 1.26, 1.29, 1.32, 1.35, 1.38, 1.41, \
        1.44, 1.47, 1.5, 1.53, 1.56, 1.59, 1.61, 1.64, 1.67, 1.69, 1.72, 1.75, 1.78, 1.81, 1.85, 1.89, 1.93, 1.97, 2.02, 2.08, \
            2.07, 2.08, 2.08, 2.08, 2.06, 2.03, 2, 1.96, 1.92, 1.87, 1.83, 1.78, 1.74, 1.69, 1.66, 1.62, 1.6, 1.57, 1.56, \
                1.54, 1.54, 1.54, 1.54, 1.54, 1.55, 1.56, 1.57, 1.59, 1.6, 1.61, 1.62, 1.63, 1.64, 1.64, 1.64, 1.64, 1.64, \
                    1.64, 1.64, 1.63, 1.64, 1.64, 1.68, 1.7, 1.73, 1.74, 1.75, 1.77, 1.8, 1.85, 1.91, 2, 2.1, 2.23, 2.39, 2.58, \
                        2.79, 3.02, 3.28, 3.56, 3.85, 4.14, 4.43, 4.71, 4.98, 5.24, 5.49, 5.74, 6, 6.27, 6.56, 6.87])
    
    return population*0.01*infected_percentage


def generated_data_sirhd(model_params, model_hyperparams, added_noise_f=0.05):
    '''
        model_params = (b, g, h, d, stochastic_noise_f)
        state_params = (population, infected, nat_imm_rate, days) 
    '''
    model_params_no_noise = list(model_params).copy()
    model_params_no_noise[-1] = 0
    # Generated data with no noise
    generated_model_no_noise = generate_data_sirhd(model_params_no_noise, model_hyperparams)[0]

    # Generated data with same params and hyperparams as above, however, with noise (stochastic + added)
    generated_model_noise = generate_data_sirhd(model_params, model_hyperparams)[0]
    generated_model_noise = np.random.normal(generated_model_noise, added_noise_f*generated_model_noise)
    stds = (added_noise_f+model_params[-1])*generated_model_no_noise  # Pseudo-backprogpagation of stds (from noise)
    weights = np.array([np.std(compartment) for compartment in generated_model_noise])
    weights = (1/np.sum(weights))*weights
    
    return generated_model_no_noise, generated_model_noise, stds, weights


def mcmc(model_params, model_hyperparams, real_dt, temperature=1, iterations=2000, chains=4, \
    proposal_width_fraction=0.5, steps_to_update=100, depth_cov_mat=50):
    
    initial_params = np.full((chains, len(model_params[:-1])), model_params[:-1])  # instantiate set of parameters per chain
    old_params = np.random.uniform(0.5*initial_params, 2*initial_params)  # initialise set of parameters; following a uniform distribution
    first_params = old_params.copy()
    old_results = np.array([run_model('sirhd', params, model_hyperparams) for params in old_params])
    S_old, I_old, R_old, H_old, D_old = old_results[:,0], old_results[:,1], old_results[:,2], old_results[:,3], old_results[:,4]

    stds = generated_data_sirhd(model_params, model_hyperparams)[2][1]
    old_chi = [chi_sq(real_dt, i, stds) for i in I_old]
    
    accepted_params = [[[o_p, o_c]] for o_p, o_c in zip(old_params, old_chi)]

    min_acc_steps = np.zeros((chains,))
    eigenvecs = None
    for i in range(iterations):
        if (1+min(min_acc_steps))%steps_to_update == 0:
            params = []  # An array to hold vectors of values for each parameter from all chains
            for j in range(first_params.shape[1]):  # Looping through all parameters
                tmp = []
                for c in range(chains):  # Looping through all chains to stack values for each parameter from all chains
                    tmp.append([accepted_params[c][i][0][j] for i in range(-int((1+min(min_acc_steps))/steps_to_update)*depth_cov_mat, 0)])
                tmp = np.array(tmp)
                params.append(tmp.flatten())
            params = np.array(params)
            
            # Obtaining the covariance matrix (bias means we use population variance equation, divide by N)
            covariance_matrix = np.cov(params, bias=True)
            eigenvals, eigenvecs = np.linalg.eig(covariance_matrix)  # Obtaining the eigenvalues and eigenvectors of the covariance matrix
            eigenvecs = None
            
        if not isinstance(eigenvecs, np.ndarray):
            new_params = abs(np.random.normal(old_params, proposal_width_fraction*initial_params))  # next guess is Gaussian centred at old_params
        else:
            # Taking random steps along the eigenvectors, scaled by the corresponding eigenvalues for each eigenvector
            steps = np.random.normal(0, np.sqrt(eigenvals)).reshape(-1, 1)
            param_steps = eigenvecs @ np.diag(eigenvals) @ steps
            param_steps = param_steps.reshape(param_steps.shape[0],)
            print(param_steps)
            new_params = list(abs(old_params + param_steps))
            print(new_params)

        new_results = np.array([run_model('sirhd', params, model_hyperparams) for params in new_params])
        S_new, I_new, R_new, H_new, D_new = new_results[:,0], new_results[:,1], new_results[:,2], new_results[:,3], new_results[:,4]
        new_chi = [chi_sq(real_dt, i, stds) for i in I_new]
        
        for chain in range(chains):
            if new_chi[chain] < old_chi[chain]:
                accepted_params[chain].append([new_params[chain], new_chi[chain]])
                old_params[chain] = new_params[chain]
                old_chi[chain] = new_chi[chain]
                min_acc_steps[chain] += 1
            else:
                if np.random.binomial(1, np.exp((old_chi[chain]-new_chi[chain])/temperature)) == 1:
                    accepted_params[chain].append([new_params[chain], new_chi[chain]])
                    old_params[chain] = new_params[chain]
                    old_chi[chain] = new_chi[chain]
                    min_acc_steps[chain] += 1
                else:
                    pass

        print('Done:', i, '/' + str(iterations), '  ||  Accepted:', min_acc_steps)
    
    for chain in range(chains):
        with open('Chains/chain_'+str(chain+1)+'.txt', 'w') as file:
            for i, step in enumerate(accepted_params[chain]):
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
    parser.add_argument('-w', '--proposal_width_fraction', type=float, help='Standard deviation of search, as a fraction of value')
    parser.add_argument('-c', '--chains', type=int, help='Number of chains for search')
    parser.add_argument('-i', '--iterations', type=int, help='Number of iterations for search')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature of Boltzmann term')
    parser.add_argument('-su', '--steps_to_update', type=int, help='how many steps before re-updating covariance matrix')
    parser.add_argument('-dc', '--depth_cov', type=int, help='Depth for covariance matrix - number of accepted points per chain')
    args = parser.parse_args()  # Parses all arguments provided at script on command-line
    
    if args.proposal_width_fraction == None:
        args.proposal_width_fraction = 0.5
    if args.chains == None:
        args.chains = 8
    if args.iterations == None:
        args.iterations = 10000
    if args.temperature == None:
        args.temperature = 1
    if args.steps_to_update == None:
        args.steps_to_update == 100
    if args.depth_cov == None:
        args.depth_cov == 50
    
    with open('model_params.txt', 'r') as file:
        lines = file.readlines()
    model_params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]

    with open('hyperparams.txt', 'r') as file:
        lines = file.readlines()
    hyperparams = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]
    hyperparams[-1] = int(hyperparams[-1])

    real_dt = real_data()
    accepted_params, first_params = mcmc(model_params, hyperparams, real_dt, temperature=args.temperature, \
        iterations=args.iterations, chains=args.chains, proposal_width_fraction=args.proposal_width_fraction, \
            steps_to_update=args.steps_to_update, depth_cov_mat=args.depth_cov)
    number_of_acc_steps = [len(a) for a in accepted_params]
    proportion_accepted = [round(n/args.iterations*100, 2) for n in number_of_acc_steps]
    print(str('%') + ' of points accepted: ', proportion_accepted)

    np.save(os.path.join('ExperimentData', 'model_params'), np.array(model_params))
    np.save(os.path.join('ExperimentData', 'hyperparams'), np.array(hyperparams))
    np.save(os.path.join('ExperimentData', 'accepted_params'), np.array(accepted_params))
    np.save(os.path.join('ExperimentData', 'first_params'), np.array(first_params))
    np.save(os.path.join('ExperimentData', 'number_of_acc_steps'), np.array(number_of_acc_steps))
