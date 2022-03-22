import matplotlib
import matplotlib.pyplot as plt
import numpy as np

gen_noise = np.load('ToPlotMyself/gen_noise.npy', allow_pickle=True)
gen_no_noise = np.load('ToPlotMyself/gen_no_noise.npy', allow_pickle=True)
infected_best = np.load('ToPlotMyself/infected_best.npy', allow_pickle=True)

fig, ax = plt.subplots()
params = {'legend.fontsize': 12}
plt.rcParams.update(params)
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'stix'

t = np.linspace(0, len(gen_noise)-1, len(gen_noise))
plt.plot(t, gen_noise, 'o', label='Simulated Data with Noise')
plt.plot(t, gen_no_noise, label='Simulated Data Underlying Signal', linewidth=3)
plt.plot(t, infected_best, label='MCMC Result using Best-Estimate ' + r'$\theta$', linewidth=3)

plt.legend(loc='upper left')
plt.xlabel('Time [days]', fontname='Times New Roman', fontsize=17)
plt.ylabel('Number of People Infected', fontname='Times New Roman', fontsize=17)
plt.minorticks_on()
ax.tick_params(direction='in')
ax.tick_params(which='minor', direction='in')
plt.xticks(fontsize=16, fontname='Times New Roman')
plt.yticks(fontsize=16, fontname='Times New Roman')
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
plt.show()