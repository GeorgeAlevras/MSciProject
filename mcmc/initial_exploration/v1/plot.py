import numpy as np
from run_model import run_model
import matplotlib.pyplot as plt

days = 100
compartments = run_model('sirv', (0.6, 0.5, 0.1), (1000e3, 1e3, 0, 0, days))[0]
S, I, R, V = compartments
N = S+I+R+V

plt.figure(1)
t_span = np.array([0, days])
t = np.linspace(t_span[0], t_span[1], t_span[1] + 1)
plt.plot(t, S/N, label='Susceptible')
plt.plot(t, I/N, label='Infected')
plt.plot(t, R/N, label='Recovered')
plt.plot(t, V/N, label='Vaccinated')
plt.xlabel('Time [days]')
plt.ylabel('Fraction of population')
plt.legend()
plt.show()
