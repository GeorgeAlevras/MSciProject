import numpy as np
from generate_data import generate_data_sirhd
from run_model import run_model
import argparse
import os
import matplotlib.pyplot as plt


def chi_sq(data, model, std_s):
    # Reduced chi-squared of model given some data
    chi_sq = []
    for i in range(len(data)):
        if std_s[i] != 0:
            chi_sq.append((data[i] - model[i])**2 / std_s[i]**2)
    
    return (1/len(chi_sq))*np.sum(chi_sq)
    # return (((data - model)**2) / std_s**2).mean()


def generated_data_sirhd(model_params, state_params, added_noise_f=0.05):
    '''
        model_params = (b, g, h, d, stochastic_noise_f)
        state_params = (population, infected, nat_imm_rate, days) 
    '''
    model_params_no_noise = list(model_params).copy()
    model_params_no_noise[-1] = 0
    # Generated data with no noise
    generated_model_no_noise = generate_data_sirhd(model_params_no_noise, state_params)[0]

    # Generated data with same params and hyperparams as above, however, with noise (stochastic + added)
    generated_model_noise = generate_data_sirhd(model_params, state_params)[0]
    generated_model_noise = np.random.normal(generated_model_noise, added_noise_f*generated_model_noise)
    stds = (added_noise_f+model_params[-1])*generated_model_no_noise  # Pseudo-backprogpagation of stds (from noise)
    # Weighing each compartment by its total variance - intended for a weighted aggregate chi-squared
    weights = np.array([np.std(compartment) for compartment in generated_model_noise])
    weights = (1/np.sum(weights))*weights
    
    return generated_model_no_noise, generated_model_noise, stds, weights


def mcmc(model_params, model_hyperparams, generated_data, temperature=1, iterations=2000, chains=4, proposal_width_fraction=0.5):
    initial_params = np.full((chains, len(model_params[:-1])), model_params[:-1])  # instantiate set of parameters per chain
    old_params = np.random.uniform(0.5*initial_params, 2*initial_params)  # initialise set of parameters; following a uniform distribution
    first_params = old_params.copy()
    old_results = np.array([run_model('sirhd', params, model_hyperparams) for params in old_params])
    S_old, I_old, R_old, H_old, D_old = old_results[:,0], old_results[:,1], old_results[:,2], old_results[:,3], old_results[:,4]
    data, std, w = generated_data
    
    old_chi = [w[0]*chi_sq(data[0], s, std[0]) + w[1]*chi_sq(data[1], i, std[1]) + w[2]*chi_sq(data[2], r, std[2]) + \
        w[3]*chi_sq(data[3], h, std[3]) + w[4]*chi_sq(data[4], d, std[4]) for s, i, r, h, d in zip(S_old, I_old, R_old, H_old, D_old)]
    accepted_params = [[[o_p, o_c]] for o_p, o_c in zip(old_params, old_chi)]

    for i in range(iterations):
        new_params = abs(np.random.normal(old_params, proposal_width_fraction*initial_params))  # next guess is Gaussian centred at old_params
        new_results = np.array([run_model('sirhd', params, model_hyperparams) for params in new_params])
        S_new, I_new, R_new, H_new, D_new = new_results[:,0], new_results[:,1], new_results[:,2], new_results[:,3], new_results[:,4]
        new_chi = [w[0]*chi_sq(data[0], s, std[0]) + w[1]*chi_sq(data[1], i, std[1]) + w[2]*chi_sq(data[2], r, std[2]) + \
        w[3]*chi_sq(data[3], h, std[3]) + w[4]*chi_sq(data[4], d, std[4]) for s, i, r, h, d in zip(S_new, I_new, R_new, H_new, D_new)]

        for chain in range(chains):
            if new_chi[chain] < old_chi[chain]:
                accepted_params[chain].append([new_params[chain], new_chi[chain]])
                old_params[chain] = new_params[chain]
                old_chi[chain] = new_chi[chain]
            else:
                if np.random.binomial(1, np.exp((old_chi[chain]-new_chi[chain])/temperature)) == 1:
                    accepted_params[chain].append([new_params[chain], new_chi[chain]])
                    old_params[chain] = new_params[chain]
                    old_chi[chain] = new_chi[chain]
                else:
                    pass
        
        if min([len(accepted_params[c]) for c in range(chains)]) > 10:
            param_1_tmp = []
            for c in range(chains):
                param_1_tmp.append([accepted_params[c][i][0][0] for i in range(len(accepted_params[c]))])
            param_1 = list(np.hstack(param_1_tmp))
            param_2_tmp = []
            for c in range(chains):
                param_2_tmp.append([accepted_params[c][i][0][1] for i in range(len(accepted_params[c]))])
            param_2 = list(np.hstack(param_2_tmp))
            param_3_tmp = []
            for c in range(chains):
                param_3_tmp.append([accepted_params[c][i][0][2] for i in range(len(accepted_params[c]))])
            param_3 = list(np.hstack(param_3_tmp))
            param_4_tmp = []
            for c in range(chains):
                param_4_tmp.append([accepted_params[c][i][0][3] for i in range(len(accepted_params[c]))])
            param_4 = list(np.hstack(param_4_tmp))

            covariance_matrix = np.cov([param_1, param_2, param_3, param_4], bias=True)
        
        print('Done: ', i, '/'+str(iterations))
    print(covariance_matrix)
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
    parser.add_argument('-d', '--days', type=int, help='Number of days of model to run')
    parser.add_argument('-w', '--proposal_width_fraction', type=float, help='Standard deviation of search, as a fraction of value')
    parser.add_argument('-c', '--chains', type=int, help='Number of chains for search')
    parser.add_argument('-i', '--iterations', type=int, help='Number of iterations for search')
    parser.add_argument('-t', '--temperature', type=float, help='Temperature of Boltzmann term')
    args = parser.parse_args()  # Parses all arguments provided at script on command-line
    
    if args.days == None:
        args.days = 100
    if args.proposal_width_fraction == None:
        args.proposal_width_fraction = 0.5
    if args.chains == None:
        args.chains = 8
    if args.iterations == None:
        args.iterations = 10000
    if args.temperature == None:
        args.temperature = 1

    with open('model_params.txt', 'r') as file:
        lines = file.readlines()
    model_params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]

    with open('hyperparams.txt', 'r') as file:
        lines = file.readlines()
    hyperparams = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]
    hyperparams[-1] = int(hyperparams[-1])

    generated_data_no_noise, generated_data_noise, generated_data_stds, generated_data_weights = generated_data_sirhd(model_params, hyperparams)
    generated_data = [generated_data_noise, generated_data_stds, generated_data_weights]
    accepted_params, first_params = mcmc(model_params, hyperparams, generated_data, temperature=args.temperature, \
        iterations=args.iterations, chains=args.chains, proposal_width_fraction=args.proposal_width_fraction)
    number_of_acc_steps = [len(a) for a in accepted_params]
    proportion_accepted = [round(n/args.iterations*100, 2) for n in number_of_acc_steps]
    print(str('%') + ' of points accepted: ', proportion_accepted)

    np.save(os.path.join('ExperimentData', 'model_params'), np.array(model_params))
    np.save(os.path.join('ExperimentData', 'hyperparams'), np.array(hyperparams))
    np.save(os.path.join('ExperimentData', 'generated_data_no_noise'), np.array(generated_data_no_noise))
    np.save(os.path.join('ExperimentData', 'generated_data_noise'), np.array(generated_data_noise))
    np.save(os.path.join('ExperimentData', 'generated_data_stds'), np.array(generated_data_stds))
    np.save(os.path.join('ExperimentData', 'generated_data_weights'), np.array(generated_data_weights))
    np.save(os.path.join('ExperimentData', 'accepted_params'), np.array(accepted_params))
    np.save(os.path.join('ExperimentData', 'first_params'), np.array(first_params))
    np.save(os.path.join('ExperimentData', 'number_of_acc_steps'), np.array(number_of_acc_steps))
