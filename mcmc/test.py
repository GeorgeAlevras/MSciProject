from run_model import run_model
from generate_data import generate_data
import numpy as np
import matplotlib.pyplot as plt

R_0 = 3.5
vaccination_rate = 0.006
vac_ef = 0.8
days = 100
std_f = 0.2

t_span = np.array([0, days])
t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)

model_params = (R_0*(1/6), R_0*(1/6), vaccination_rate, 0.083333333, 0.15, 0.1, (1/6), (1/6), 0.075, 0.003428571, 0.025, (1-vac_ef)*R_0*(1/6), (1-vac_ef)*R_0*(1/6), 0.1, 0.00035, 0.1, 0.00125, 0.2, (1/6), std_f)
model_state_params = [1000e3, 1e3, 0, 0, days]
generated_model, N = generate_data(model_params, model_state_params)
generated_model_infected_symptomatic = generated_model[3] + generated_model[9]
generated_model_infected_symptomatic = np.random.normal(generated_model_infected_symptomatic, 0.05*generated_model_infected_symptomatic)
plt.plot(t, generated_model_infected_symptomatic/N, '.', label='Generated Data', color='black')

solutions = []
arguments = []
model_params_c = np.array((R_0*(1/6), R_0*(1/6), vaccination_rate, 0.083333333, 0.15, 0.1, (1/6), (1/6), 0.075, 0.003428571, 0.025, (1-vac_ef)*R_0*(1/6), (1-vac_ef)*R_0*(1/6), 0.1, 0.00035, 0.1, 0.00125, 0.2, (1/6)))
for i in range(1000):
    model_params = abs(np.random.normal(model_params_c, 0.1*model_params_c))
    model_results = run_model('sirv_c', model_params, model_state_params)
    solutions.append(np.array(model_results[0]))
    arguments.append(model_params)
    S, E, A, Sy, H, R, D, V, A_v, Sy_v, H_v = solutions[i]
    N = S + E + A + Sy + H + R + V + A_v + Sy_v + H_v
    plt.plot(t, (Sy+Sy_v)/N, alpha=0.1, color='green')
    print('Done:', i+1,'/1000')

plt.show()   
