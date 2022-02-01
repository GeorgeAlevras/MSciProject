import numpy as np
from generate_data import generate_data_seasyhrva_vsy_vh_vd
from run_model import run_model
import argparse
import os


def chi_sq(expected, observed):
    return np.sum([((e-o)**2)/(2*np.std(expected)**2) for e, o in zip(expected, observed)])


def generated_data_seasyhrva_vsy_vh_vd(model_params, state_params, added_noise_f=0.05):
    '''
        model_params = (b_a, b_sy, g_a, g_sy, g_h, e_a, e_sy, a, h, v, b_a_v, b_sy_v, g_a_v, g_sy_v, g_h_v, a_v, h_v, d_h, d_h_v, std_f)
        state_params = (population, exposed, infected, recovered, vaccinated, days) 
    '''

    model_params_no_noise = list(model_params).copy()
    model_params_no_noise[-1] = 0
    generated_model_no_noise, N_no_noise = generate_data_seasyhrva_vsy_vh_vd(model_params_no_noise, state_params)
    generated_model_noise = generate_data_seasyhrva_vsy_vh_vd(model_params, state_params)[0]
    generated_model_susceptible = np.random.normal(generated_model_noise[0], added_noise_f*generated_model_noise[0])
    generated_model_exposed = np.random.normal(generated_model_noise[1], added_noise_f*generated_model_noise[1])
    generated_model_asymptomatic = np.random.normal(generated_model_noise[2], added_noise_f*generated_model_noise[2])
    generated_model_symptomatic = np.random.normal(generated_model_noise[3], added_noise_f*generated_model_noise[3])
    generated_model_asymptomatic_vac = np.random.normal(generated_model_noise[4], added_noise_f*generated_model_noise[4])
    generated_model_symptomatic_vac = np.random.normal(generated_model_noise[5], added_noise_f*generated_model_noise[5])
    generated_model_hospitalised = np.random.normal(generated_model_noise[6], added_noise_f*generated_model_noise[6])
    generated_model_hospitalised_vac = np.random.normal(generated_model_noise[7], added_noise_f*generated_model_noise[7])
    generated_model_recovered = np.random.normal(generated_model_noise[8], added_noise_f*generated_model_noise[8])
    generated_model_vaccinated = np.random.normal(generated_model_noise[9], added_noise_f*generated_model_noise[9])
    generated_model_deceased = np.random.normal(generated_model_noise[10], added_noise_f*generated_model_noise[10])
    data_std = np.std(generated_model_susceptible) + np.std(generated_model_exposed) + np.std(generated_model_asymptomatic) + \
         np.std(generated_model_symptomatic) + np.std(generated_model_asymptomatic_vac) + np.std(generated_model_symptomatic_vac) + \
             np.std(generated_model_hospitalised) + np.std(generated_model_hospitalised_vac) + np.std(generated_model_recovered) + \
                 np.std(generated_model_vaccinated) + np.std(generated_model_deceased)

    return generated_model_no_noise, data_std, generated_model_susceptible, generated_model_exposed, generated_model_asymptomatic, \
        generated_model_symptomatic, generated_model_asymptomatic_vac, generated_model_symptomatic_vac, generated_model_hospitalised, \
            generated_model_hospitalised_vac, generated_model_recovered, generated_model_vaccinated, generated_model_deceased
    

def mcmc_seasyhrva_vsy_vh_vd(model_params, state_params, gen_susc, gen_exp, gen_asym, gen_sym, gen_asym_v, gen_sym_v, gen_hosp, gen_hosp_v, gen_rec, \
    gen_vac, gen_dec, gen_std, iterations, std_f, chains=4, burn_in=1000, p=0.2):

    initial_params = np.full((chains, len(model_params[:-1])), model_params[:-1])
    old_params = np.random.uniform(0.5*initial_params, 2*initial_params)  # initial set of parameters follows uniform dist
    first_params = old_params.copy()
    old_results = np.array([run_model('seasyhrva_vsy_vh_vd', params, state_params)[0] for params in old_params])
    old_susceptible = old_results[:,0]
    old_exposed = old_results[:,1]
    old_asymptomatic = old_results[:,2]
    old_symptomatic = old_results[:,3]
    old_asymptomatic_vac = old_results[:,4]
    old_symptomatic_vac = old_results[:,5]
    old_hospitalised = old_results[:,6]
    old_hospitalised_v = old_results[:,7]
    old_recovered = old_results[:,8]
    old_vaccinated = old_results[:,9]
    old_deceased = old_results[:,10]
    old_chi = np.array([chi_sq(gen_susc, i) + chi_sq(gen_exp, j) + chi_sq(gen_asym, k) + chi_sq(gen_sym, l) + \
        chi_sq(gen_asym_v, m) + chi_sq(gen_sym_v, n) + chi_sq(gen_hosp, o) + chi_sq(gen_hosp_v, p) + chi_sq(gen_rec, q) + \
        chi_sq(gen_vac, r) + chi_sq(gen_dec, s) for i, j, k, l, m, n, o, p, q, r, s in zip(old_susceptible, old_exposed, old_asymptomatic, \
            old_symptomatic, old_asymptomatic_vac, old_symptomatic_vac, old_hospitalised, old_hospitalised_v, old_recovered, \
                old_vaccinated, old_deceased)])
    accepted_params = [[[o_p, o_c]] for o_p, o_c in zip(old_params, old_chi)]

    counter_lower = 0
    counter_higher = 0
    for i in range(iterations):
        min_acc = min([len(a) for a in accepted_params])
        new_params = abs(np.random.normal(old_params, std_f*initial_params))  # next guess is Gaussian centred at old_params
        new_results = np.array([run_model('seasyrvd', params, state_params)[0] for params in new_params])
        new_susceptible = new_results[:,0]
        new_exposed = new_results[:,1]
        new_asymptomatic = new_results[:,2]
        new_symptomatic = new_results[:,3]
        new_asymptomatic_vac = new_results[:,4]
        new_symptomatic_vac = new_results[:,5]
        new_hospitalised = new_results[:,6]
        new_hospitalised_vac = new_results[:,7]
        new_recovered = new_results[:,8]
        new_vaccinated = new_results[:,9]
        new_deceased = new_results[:,10]
        new_chi = np.array([chi_sq(gen_susc, i) + chi_sq(gen_exp, j) + chi_sq(gen_asym, k) + chi_sq(gen_sym, l) + \
        chi_sq(gen_asym_v, m) + chi_sq(gen_sym_v, n) + chi_sq(gen_hosp, o) + chi_sq(gen_hosp_v, p) + chi_sq(gen_rec, q) + \
        chi_sq(gen_vac, r) + chi_sq(gen_dec, s) for i, j, k, l, m, n, o, p, q, r, s in zip(new_susceptible, new_exposed, new_asymptomatic, \
            new_symptomatic, new_asymptomatic_vac, new_symptomatic_vac, new_hospitalised, new_hospitalised_vac, new_recovered, new_vaccinated, new_deceased)])
        for chain in range(chains):
            boltzmann_c = (-1)*gen_std*np.log(p)/(new_chi[chain]-old_chi[chain])
            if new_chi[chain] < old_chi[chain]:
                accepted_params[chain].append([new_params[chain], new_chi[chain]])
                old_params[chain] = new_params[chain]
                old_chi[chain] = new_chi[chain]
                counter_lower += 1
            else:
                if np.random.binomial(1, np.exp(-boltzmann_c*(new_chi[chain]-old_chi[chain])/gen_std)) == 1:
                    accepted_params[chain].append([new_params[chain], new_chi[chain]])
                    old_params[chain] = new_params[chain]
                    old_chi[chain] = new_chi[chain]
                    counter_higher += 1
                else:
                    pass
        if min_acc%burn_in == 0:  # every integer multiple of burn-in time
            std_f *= 1
            p *= 1
        print('Done: ', i, '/'+str(iterations), ' Accepted:', len(accepted_params[0]), 'Acc Low:', int(counter_lower/chains), 'Acc High:', int(counter_higher/chains))
    
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
    parser.add_argument('-p', '--probability', type=float, help='Probability of choosing values with lower chi-square')
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
    if args.probability == None:
        args.probability = 0.08
    
    with open('model_params.txt', 'r') as file:
        lines = file.readlines()
    model_params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]

    with open('state_params.txt', 'r') as file:
        lines = file.readlines()
    state_params = [float(lines[i].rstrip('\n').split(' ')[-1]) for i in range(len(lines))]
    state_params[-1] = int(state_params[-1])

    generated_model_no_noise, gen_std, generated_model_susceptible, generated_model_exposed, generated_model_asymptomatic, \
        generated_model_symptomatic, generated_model_asymptomatic_vac, generated_model_symptomatic_vac, generated_model_hospitalised, \
            generated_model_hospitalised_vac, generated_model_recovered, generated_model_vaccinated, generated_model_deceased = generated_data_seasyhrva_vsy_vh_vd(model_params, state_params)
    accepted, first = mcmc_seasyhrva_vsy_vh_vd(model_params, state_params, generated_model_susceptible, generated_model_exposed, \
        generated_model_asymptomatic, generated_model_symptomatic, generated_model_asymptomatic_vac, generated_model_symptomatic_vac, generated_model_hospitalised, generated_model_hospitalised_vac, \
            generated_model_recovered, generated_model_vaccinated, generated_model_deceased, gen_std, iterations=args.iterations, std_f=args.std, chains=args.chains, burn_in=args.burn_in, p=args.probability)
    number_of_acc_steps = [len(accepted[i]) for i in range(len(accepted))]

    np.save(os.path.join('ExperimentData', 'generated_model_no_noise'), np.array(generated_model_no_noise))
    np.save(os.path.join('ExperimentData', 'gen_std'), np.array(gen_std))
    np.save(os.path.join('ExperimentData', 'generated_model_susceptible'), np.array(generated_model_susceptible))
    np.save(os.path.join('ExperimentData', 'generated_model_exposed'), np.array(generated_model_exposed))
    np.save(os.path.join('ExperimentData', 'generated_model_asymptomatic'), np.array(generated_model_asymptomatic))
    np.save(os.path.join('ExperimentData', 'generated_model_symptomatic'), np.array(generated_model_symptomatic))
    np.save(os.path.join('ExperimentData', 'generated_model_asymptomatic_vac'), np.array(generated_model_asymptomatic_vac))
    np.save(os.path.join('ExperimentData', 'generated_model_symptomatic_vac'), np.array(generated_model_symptomatic_vac))
    np.save(os.path.join('ExperimentData', 'generated_model_hospitalised'), np.array(generated_model_hospitalised))
    np.save(os.path.join('ExperimentData', 'generated_model_hospitalised_vac'), np.array(generated_model_hospitalised_vac))
    np.save(os.path.join('ExperimentData', 'generated_model_recovered'), np.array(generated_model_recovered))
    np.save(os.path.join('ExperimentData', 'generated_model_vaccinated'), np.array(generated_model_vaccinated))
    np.save(os.path.join('ExperimentData', 'generated_model_deceased'), np.array(generated_model_deceased))
    np.save(os.path.join('ExperimentData', 'accepted'), np.array(accepted))
    np.save(os.path.join('ExperimentData', 'first'), np.array(first))
    np.save(os.path.join('ExperimentData', 'model_params'), np.array(model_params))
    np.save(os.path.join('ExperimentData', 'state_params'), np.array(state_params))
    np.save(os.path.join('ExperimentData', 'number_of_acc_steps'), np.array(number_of_acc_steps))
