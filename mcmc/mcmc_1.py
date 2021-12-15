import numpy as np
from generate_data import generate_data
from run_model import run_model
import argparse
import matplotlib.pyplot as plt
import time

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
    model_params_old_1 = abs(np.random.normal(init_params, 0.3*np.array(init_params)))
    model_results_old_1 = np.array(run_model('sirv_c', model_params_old_1, model_state_params)[0])
    infected_old_1 = model_results_old_1[3] + model_results_old_1[9]
    chi_old_1 = chi_sq(generated_model_infected_symptomatic, infected_old_1)
    accepted_params_1 = [[model_params_old_1.tolist(), chi_old_1]]
    
    model_params_old_2 = abs(np.random.normal(init_params, 0.3*np.array(init_params)))
    model_results_old_2 = np.array(run_model('sirv_c', model_params_old_2, model_state_params)[0])
    infected_old_2 = model_results_old_2[3] + model_results_old_2[9]
    chi_old_2 = chi_sq(generated_model_infected_symptomatic, infected_old_2)
    accepted_params_2 = [[model_params_old_2.tolist(), chi_old_2]]
    
    model_params_old_3 = abs(np.random.normal(init_params, 0.3*np.array(init_params)))
    model_results_old_3 = np.array(run_model('sirv_c', model_params_old_3, model_state_params)[0])
    infected_old_3 = model_results_old_3[3] + model_results_old_3[9]
    chi_old_3 = chi_sq(generated_model_infected_symptomatic, infected_old_3)
    accepted_params_3 = [[model_params_old_3.tolist(), chi_old_3]]
    
    model_params_old_4 = abs(np.random.normal(init_params, 0.3*np.array(init_params)))
    model_results_old_4 = np.array(run_model('sirv_c', model_params_old_4, model_state_params)[0])
    infected_old_4 = model_results_old_4[3] + model_results_old_4[9]
    chi_old_4 = chi_sq(generated_model_infected_symptomatic, infected_old_4)
    accepted_params_4 = [[model_params_old_4.tolist(), chi_old_4]]

    model_params_old_5 = abs(np.random.normal(init_params, 0.3*np.array(init_params)))
    model_results_old_5 = np.array(run_model('sirv_c', model_params_old_5, model_state_params)[0])
    infected_old_5 = model_results_old_5[3] + model_results_old_5[9]
    chi_old_5 = chi_sq(generated_model_infected_symptomatic, infected_old_5)
    accepted_params_5 = [[model_params_old_5.tolist(), chi_old_5]]
    
    model_params_old_6 = abs(np.random.normal(init_params, 0.3*np.array(init_params)))
    model_results_old_6 = np.array(run_model('sirv_c', model_params_old_6, model_state_params)[0])
    infected_old_6 = model_results_old_6[3] + model_results_old_6[9]
    chi_old_6 = chi_sq(generated_model_infected_symptomatic, infected_old_6)
    accepted_params_6 = [[model_params_old_6.tolist(), chi_old_6]]

    for i in range(50000):
        model_params_new_1 = abs(np.random.normal(model_params_old_1, args.std*np.array(model_params_old_1)))
        model_results_new_1 = np.array(run_model('sirv_c', model_params_new_1, model_state_params)[0])
        infected_new_1 = model_results_new_1[3] + model_results_new_1[9]
        chi_new_1 = chi_sq(generated_model_infected_symptomatic, infected_new_1)
        if chi_new_1 < chi_old_1:
            accepted_params_1.append([model_params_new_1.tolist(), chi_new_1])
            model_params_old_1 = model_params_new_1
            chi_old_1 = chi_new_1
        else:
            pass
        
        model_params_new_2 = abs(np.random.normal(model_params_old_2, args.std*np.array(model_params_old_2)))
        model_results_new_2 = np.array(run_model('sirv_c', model_params_new_2, model_state_params)[0])
        infected_new_2 = model_results_new_2[3] + model_results_new_2[9]
        chi_new_2 = chi_sq(generated_model_infected_symptomatic, infected_new_2)
        if chi_new_2 < chi_old_2:
            accepted_params_2.append([model_params_new_2.tolist(), chi_new_2])
            model_params_old_2 = model_params_new_2
            chi_old_2 = chi_new_2
        else:
            pass

        model_params_new_3 = abs(np.random.normal(model_params_old_3, args.std*np.array(model_params_old_3)))
        model_results_new_3 = np.array(run_model('sirv_c', model_params_new_3, model_state_params)[0])
        infected_new_3 = model_results_new_3[3] + model_results_new_3[9]
        chi_new_3 = chi_sq(generated_model_infected_symptomatic, infected_new_3)
        if chi_new_3 < chi_old_3:
            accepted_params_3.append([model_params_new_3.tolist(), chi_new_3])
            model_params_old_3 = model_params_new_3
            chi_old_3 = chi_new_3
        else:
            pass

        model_params_new_4 = abs(np.random.normal(model_params_old_4, args.std*np.array(model_params_old_4)))
        model_results_new_4 = np.array(run_model('sirv_c', model_params_new_4, model_state_params)[0])
        infected_new_4 = model_results_new_4[3] + model_results_new_4[9]
        chi_new_4 = chi_sq(generated_model_infected_symptomatic, infected_new_4)
        if chi_new_4 < chi_old_4:
            accepted_params_4.append([model_params_new_4.tolist(), chi_new_4])
            model_params_old_4 = model_params_new_4
            chi_old_4 = chi_new_4
        else:
            pass

        model_params_new_5 = abs(np.random.normal(model_params_old_5, args.std*np.array(model_params_old_5)))
        model_results_new_5 = np.array(run_model('sirv_c', model_params_new_5, model_state_params)[0])
        infected_new_5 = model_results_new_5[3] + model_results_new_5[9]
        chi_new_5 = chi_sq(generated_model_infected_symptomatic, infected_new_5)
        if chi_new_5 < chi_old_5:
            accepted_params_5.append([model_params_new_5.tolist(), chi_new_5])
            model_params_old_5 = model_params_new_5
            chi_old_5 = chi_new_5
        else:
            pass

        model_params_new_6 = abs(np.random.normal(model_params_old_6, args.std*np.array(model_params_old_6)))
        model_results_new_6 = np.array(run_model('sirv_c', model_params_new_6, model_state_params)[0])
        infected_new_6 = model_results_new_6[3] + model_results_new_6[9]
        chi_new_6 = chi_sq(generated_model_infected_symptomatic, infected_new_6)
        if chi_new_6 < chi_old_6:
            accepted_params_6.append([model_params_new_6.tolist(), chi_new_6])
            model_params_old_6 = model_params_new_6
            chi_old_6 = chi_new_6
        else:
            pass
        print('Done: ', i, '/50000')
    print('No. of steps accepted: ', len(accepted_params_1))
    # a = np.array(accepted_params_1)[-5:,0]
    # averages = [np.average([a[0][i], a[1][i], a[2][i], a[3][i], a[4][i]]) for i in range(len(a[0]))]
    print(round(accepted_params_1[-1][1]/accepted_params_1[0][1], 3))
    end_params = np.mean([accepted_params_1[-1][0], accepted_params_2[-1][0], accepted_params_3[-1][0], accepted_params_4[-1][0], accepted_params_5[-1][0], accepted_params_6[-1][0]], axis=0)
    print('Final parameters:', end_params)
    # print('Average of last 5 sets of parameters:', averages)
    
    # with open('Files/mcmc.txt', 'w') as file:
    #     for i, step in enumerate(accepted_params):
    #         file.write(str(i)+','+str(step[1])+',')
    #         for j, param in enumerate(step[0]):
    #             file.write(str(param))
    #             if j != len(step[0])-1:
    #                 file.write(',')
    #         file.write('\n')
    

    X = ['R_0', 'b_a', 'b_sy', 'v', 'e_a', 'e_sy', 'a', 'g_a', 'g_sy', 'g_h', 'h', 'd_h', 'b_a_v', 'b_sy_v', 'a_v', 'h_v', 'g_h_v', 'd_h_v', 'g_a_v', 'g_sy_v']
    X_axis = np.arange(len(X))
    model_params = list(model_params)
    model_params.insert(0, 3.5)
    # r_0 = (accepted_params[-1][0][0]+accepted_params[-1][0][1])/(accepted_params[-1][0][6]+accepted_params[-1][0][7])
    r_0 = (end_params[0]+end_params[1])/(end_params[6]+end_params[7])
    # mcmc_params = list(accepted_params[-1][0])
    mcmc_params = list(end_params)
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
