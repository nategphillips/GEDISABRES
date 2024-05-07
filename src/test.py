# module test
'''
Model equilibrium tranmission using the Beer-Lambert law.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots  # pylint: disable = unused-import

import constants as cn

plt.style.use(['science', 'grid'])


def calc_density_cm(pressure_torr, temp):
    pressure_pa = pressure_torr * 101325 / 760

    return pressure_pa / (cn.BOLTZ * temp) / 1e6


vib_qn_up = 0
length = 2    # [cm]
temperature = 3000  # [K]
pressure = 5    # [Torr]
density = calc_density_cm(pressure, temperature)  # [cm^-3]
density = 1

# print(f'Total number density: {density:10.4e} [cm^-3]')

df = pd.read_csv('../data/molecular_constants/o2.csv', index_col=0)
consts = df.loc['x3sg']  # ground state
fc_data = np.loadtxt('../data/franck-condon/o2.csv', delimiter=',')


def vibrational_term(vib_qn):
    return (consts['w_e'] * (vib_qn + 0.5) -
            consts['we_xe'] * (vib_qn + 0.5)**2 +
            consts['we_ye'] * (vib_qn + 0.5)**3 +
            consts['we_ze'] * (vib_qn + 0.5)**4)


def get_fc(vib_qn):
    return fc_data[vib_qn_up][vib_qn]


vibrational_levels = np.arange(0, 19)

total_partition = (np.exp(-vibrational_term(vibrational_levels) *
                          cn.PLANC * cn.LIGHT / (cn.BOLTZ * temperature))).sum()


def number_density(vib_qn):
    return (density * np.exp(-vibrational_term(vib_qn) * cn.PLANC * cn.LIGHT /
                             (cn.BOLTZ * temperature)) / total_partition)


x = np.linspace(vibrational_levels.min(), vibrational_levels.max(), 1000)


def plot_text(func):
    for level in vibrational_levels:
        plt.text(level + 0.1, func(level), f'{level}')


plt.plot(vibrational_levels, number_density(vibrational_levels), 'o')
plt.plot(x, number_density(x), label='Boltzmann Distribution')
plot_text(number_density)
plt.xlabel('Ground State Vibrational Level, $v\'\'$ [-]')
plt.ylabel('Relative Vibrational Number Density, $N_v$ [-]')
plt.legend()
plt.show()

plt.plot(vibrational_levels, get_fc(vibrational_levels), 'o')
plot_text(get_fc)
plt.xlabel('Ground State Vibrational Level, $v\'\'$ [-]')
plt.ylabel('Franck-Condon Factor, FCF [-]')
plt.show()


def factor(vib_qn):
    return get_fc(vib_qn) * number_density(vib_qn) * length


plt.plot(vibrational_levels, factor(vibrational_levels), 'o')
plot_text(factor)
plt.xlabel('Ground State Vibrational Level, $v\'\'$ [-]')
plt.ylabel('Factor, $\\sigma NL$ [cm]')
plt.show()

# def transmission(vib_qn):
#     return np.exp(-factor(vib_qn))

# plt.plot(vibrational_levels, transmission(vibrational_levels), 'o')
# plot_text(transmission)
# plt.xlabel('Ground State Vibrational Level, $v\'\'$ [-]')
# plt.ylabel('Transmission, $T$ [-]')
# plt.show()
