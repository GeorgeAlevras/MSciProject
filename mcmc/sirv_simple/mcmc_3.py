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
    generated_model_infected = np.random.normal(generated_model_noise[1], added_noise_f*generated_model_noise[1])
    generated_model_susceptible = np.random.normal(generated_model_noise[0], added_noise_f*generated_model_noise[0])
    generated_model_recovered = np.random.normal(generated_model_noise[2], added_noise_f*generated_model_noise[2])
    generated_model_vaccinated = np.random.normal(generated_model_noise[3], added_noise_f*generated_model_noise[3])
    data_std = np.std(generated_model_infected)

    return generated_model_no_noise, generated_model_infected, data_std, generated_model_susceptible, generated_model_recovered, generated_model_vaccinated
    

def mcmc(model_params, state_params, gen_infected, gen_susc, gen_rec, gen_vac, gen_std, iterations, std_f, chains=4, burn_in=1000):
    initial_params = np.full((chains, len(model_params[:-1])), model_params[:-1])
    old_params = np.random.uniform(0.5*initial_params, 2*initial_params)  # initial set of parameters follows uniform dist
    first_params = old_params.copy()
    old_results = np.array([run_model('sirv', params, state_params)[0] for params in old_params])
    old_infected = old_results[:,1]
    old_susceptible = old_results[:,0]
    old_recovered = old_results[:,2]
    old_vaccinated = old_results[:,3]
    old_chi = np.array([chi_sq(gen_infected, i) + chi_sq(gen_susc, j) + chi_sq(gen_rec, k) + chi_sq(gen_vac, z) for i, j, k, z in zip(old_infected, old_susceptible, old_recovered, old_vaccinated)])
    accepted_params = [[[o_p, o_c]] for o_p, o_c in zip(old_params, old_chi)]

    p = 0.2  # probability of choosing parameters with higher chi-square value
    for i in range(iterations):
        min_acc = min([len(a) for a in accepted_params])
        new_params = abs(np.random.normal(old_params, std_f*initial_params))  # next guess is Gaussian centred at old_params
        new_results = np.array([run_model('sirv', params, state_params)[0] for params in new_params])
        new_infected = new_results[:,1]
        new_susceptible = new_results[:,0]
        new_recovered = new_results[:,2]
        new_vaccinated = new_results[:,3]
        new_chi = np.array([chi_sq(gen_infected, i) + chi_sq(gen_susc, j) + chi_sq(gen_rec, k) + chi_sq(gen_vac, z) for i, j, k, z in zip(new_infected, new_susceptible, new_recovered, new_vaccinated)])
        
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
            std_f *= 1  # width of Gaussian search decreases by 30%
            p *= 1  # probability of choosing higher chi-square points decreases by 75%
        print('Done: ', i, '/'+str(iterations))
    
    for chain in range(chains):
        with open('New_Files/chain_'+str(chain+1)+'.txt', 'w') as file:
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
    if args.std == None:
        args.std = 0.5
    
    model_params = (0.583333, 0.166667, 0.006, 0.2)
    state_params = (1000e3, 1e3, 0, 0, args.days) 
    generated_model_no_noise, generated_model_infected, gen_std, generated_model_susceptible, generated_model_recovered, generated_model_vaccinated = generated_data(model_params, state_params)
    accepted, first = mcmc(model_params, state_params, generated_model_infected, generated_model_susceptible, generated_model_recovered, generated_model_vaccinated, gen_std, iterations=40000, std_f=args.std, chains=8, burn_in=1500)
    
    chi_sq_values = [accepted[i][-1][1] for i in range(len(accepted))]
    best_params = accepted[np.argmin(chi_sq_values)][-1][0]
    number_of_acc_steps = [len(accepted[i]) for i in range(len(accepted))]
    print('Number of accepted steps: ', number_of_acc_steps)
    print('Best parameters (chain '+str(np.argmin(chi_sq_values)+1)+'):', best_params)

    plt.figure(1)
    X = ['b', 'g', 'v']
    X_axis = np.arange(len(X))
    model_params = list(model_params)
    mcmc_params = list(best_params)
    plt.bar(X_axis-0.2, np.array(model_params[:-1])/np.array(model_params[:-1]), 0.2, label='Generated Data')
    plt.bar(X_axis+0.2, np.array(mcmc_params)/np.array(model_params[:-1]), 0.2, label='MCMC Data')
    plt.xticks(X_axis, X)
    plt.xlabel("Parameters")
    plt.ylabel("Parameters, normalised by generated data parameters")
    plt.title("MCMC Parameters Comparison")
    plt.legend()
    plt.savefig('Images/parameter_relative_ratios.png')

    plt.figure(2)
    t_span = np.array([0, args.days])
    t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
    plt.plot(t, generated_model_infected, '.', label='Generated Data')
    plt.plot(t, generated_model_no_noise[1], label='Underlying Gen Data Signal')
    model_results_best = np.array(run_model('sirv', best_params, state_params)[0])
    infected_best = model_results_best[1]
    plt.plot(t, infected_best, label='MCMC Result')
    plt.xlabel('time')
    plt.ylabel('Infected People')
    plt.legend()
    plt.savefig('Images/results.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Georgios Alevras: MCMC Algorithm 3',
                                     epilog='Enjoy the script :)')
    parser.add_argument('-d', '--days', type=int, help='Number of days of model to run')
    parser.add_argument('-s', '--std', type=float, help='Standard deviation of search, as a fraction of value')
    arguments = parser.parse_args()  # Parses all arguments provided at script on command-line
    use_args(arguments)  # Executes code according to arguments provided
