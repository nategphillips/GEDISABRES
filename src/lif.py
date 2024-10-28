# module lif

import matplotlib.pyplot as plt
import numpy as np
import scipy as sy

import constants as cn
import main as m


def laser_intensity(t: np.ndarray, pulse_center: float, pulse_width: float, fluence: float):
    return (
        fluence
        / pulse_width
        * np.sqrt(4 * np.log(2) / np.pi)
        * np.exp(-4 * np.log(2) * ((t - pulse_center) / pulse_width) ** 2)
    )


def rate_equations(
    n: list[float],
    t: np.ndarray,
    f_b: float,
    w_c: float,
    w_d: float,
    w_f: float,
    w_q: float,
    a_21: float,
    b_12: float,
    b_21: float,
    pulse_center: float,
    pulse_width: float,
    fluence: float,
):
    n1, n2, n3 = n

    overlap_integral: float = 3.0  # [cm]

    i_l: float = laser_intensity(t, pulse_center, pulse_width, fluence)
    w_la: float = i_l * b_12 * overlap_integral / cn.LIGHT
    w_le: float = i_l * b_21 * overlap_integral / cn.LIGHT

    dn1_dt: float = -w_la * n1 + n2 * (w_le + a_21) + w_c * (n3 - n1)
    dn2_dt: float = w_la * n1 - n2 * (w_le + w_d + a_21 + w_f + w_q)
    dn3_dt: float = -w_c * f_b / (1 - f_b) * (n3 - n1)

    return [dn1_dt, dn2_dt, dn3_dt]


def simulate(
    line, a21_coeffs, v_up, v_lo, laser_energy, laser_area, pres, temp, pulse_center, pulse_width
):
    g_l: int = 3
    g_u: int = 1
    j_qn: int = line.j_qn_lo
    f_b: float = line.rot_boltz_frac
    s_j: float = line.honl_london_factor
    nu: float = line.wavenumber  # [1/cm]
    nu_d: float = line.predissociation()  # [1/cm]
    w_d: float = 2 * np.pi * cn.LIGHT * nu_d  # [1/s]
    a_21: float = a21_coeffs[v_up][v_lo] * s_j / (2 * j_qn + 1)  # [1/s]
    w_f: float = np.sum(a21_coeffs[v_up]) * s_j / (2 * j_qn + 1)  # [1/s]
    b_12: float = a_21 / (8 * np.pi * cn.PLANC * cn.LIGHT * nu**3) * g_u / g_l  # [cm/J]
    b_21: float = b_12 * g_l / g_u  # [cm/J]

    fluence: float = laser_energy / laser_area

    # These two use pressure in atm
    w_c: float = 7.78e9 * (pres / 101325) * np.sqrt(300 / temp)  # [1/s]
    w_q: float = 7.8e9 * (pres / 101325) * np.sqrt(300 / temp)  # [1/s]

    # print(f"w_c: {w_c:.4e}")
    # print(f"w_d: {w_d:.4e}")
    # print(f"w_f: {w_f:.4e}")
    # print(f"w_q: {w_q:.4e}")
    # print(f"a_21: {a_21:.4e}")
    # print(f"b_12: {b_12:.4e}")
    # print(f"b_21: {b_21:.4e}")

    t: np.ndarray = np.linspace(0, 60e-9, 1000)
    n: list[float] = [1.0, 0.0, 1.0]

    solution = sy.integrate.odeint(
        rate_equations,
        n,
        t,
        args=(f_b, w_c, w_d, w_f, w_q, a_21, b_12, b_21, pulse_center, pulse_width, fluence),
    )

    n1 = solution[:, 0]
    n2 = solution[:, 1]
    n3 = solution[:, 2]

    s_f = sy.integrate.cumulative_trapezoid(w_f * n2, t, initial=0)

    return s_f.max()

    # s_f /= n2.max()

    # i_l = laser_intensity(t, pulse_center, pulse_width, fluence)
    # i_l /= i_l.max()

    # _, ax1 = plt.subplots()

    # ax1.set_xlabel("Time [s]")
    # ax1.set_ylabel("N1, N3, IL")
    # ax1.plot(t, n1, label="N1")
    # ax1.plot(t, n3, label="N3")
    # ax1.plot(t, i_l, label="IL")

    # ax2 = ax1.twinx()
    # ax2.set_ylabel("N2, SF")
    # ax2.plot(t, n2, label="N2")
    # ax2.plot(t, s_f, label="SF")

    # plt.legend()
    # plt.show()


def get_signal(v_up, v_lo, branch_name, branch_idx_lo, n_qn_lo, energy_range, temp):
    molecule = m.Molecule(name="O2", atom_1=m.Atom("O"), atom_2=m.Atom("O"))

    state_up = m.ElectronicState(name="B3Su-", spin_multiplicity=3, molecule=molecule)
    state_lo = m.ElectronicState(name="X3Sg-", spin_multiplicity=3, molecule=molecule)

    # User settings
    pres = 101325.0  # [Pa]
    pulse_center: float = 30e-9  # [s]
    pulse_width: float = 20e-9  # [s]
    laser_area: float = 1.0  # [cm^2]

    vib_bands: list[tuple[int, int]] = [(v_up, v_lo)]

    sim = m.Simulation(
        sim_type=m.SimulationType.ABSORPTION,
        molecule=molecule,
        state_up=state_up,
        state_lo=state_lo,
        rot_lvls=np.arange(0, 40),
        temp_trn=temp,
        temp_elc=temp,
        temp_vib=temp,
        temp_rot=temp,
        pressure=pres,
        vib_bands=vib_bands,
    )

    for desired_line in sim.vib_bands[0].rot_lines:
        if (
            desired_line.branch_name == branch_name
            and desired_line.branch_idx_lo == branch_idx_lo
            and desired_line.n_qn_lo == n_qn_lo
            and not desired_line.is_satellite
        ):
            line = desired_line

    a21_coeffs = np.loadtxt(
        f"../data/{molecule.name}/einstein/{state_up.name}_to_{state_lo.name}_allison.csv",
        delimiter=",",
    )

    signals = []

    for laser_energy in energy_range:
        signal = simulate(
            line,
            a21_coeffs,
            v_up,
            v_lo,
            laser_energy,
            laser_area,
            pres,
            temp,
            pulse_center,
            pulse_width,
        )
        signals.append(signal)

    return signals


def main() -> None:
    # 1800 K
    jay_27_p9x = np.array([0, 1.8, 3.6, 6, 12, 24, 42.5]) / 1e3
    jay_27_p9y = np.array([0, 0.08, 0.15, 0.27, 0.47, 0.7, 1])
    plt.scatter(jay_27_p9x, jay_27_p9y)
    energy1 = np.linspace(jay_27_p9x[0], jay_27_p9x[-1])
    signal1 = get_signal(2, 7, "P", 1, 9, energy1, 1800)
    plt.plot(energy1, signal1 / max(signal1))

    # 1800 K
    jay_06_r17x = np.array([0, 2, 3.8, 7, 12.1, 23, 43]) / 1e3
    jay_06_r17y = np.array([0, 0.025, 0.06, 0.12, 0.27, 0.55, 1])
    plt.scatter(jay_06_r17x, jay_06_r17y)
    energy2 = np.linspace(jay_06_r17x[0], jay_06_r17x[-1])
    signal2 = get_signal(0, 6, "R", 1, 17, energy2, 1800)
    plt.plot(energy2, signal2 / max(signal2))

    # # 1475 K
    # jay_06_r17x2 = np.array([0, 60, 75, 95, 125, 525, 625, 750, 860, 1020]) / 1e3
    # jay_06_r17y2 = np.array([0, 0.28, 0.35, 0.32, 0.54, 1.62, 1.86, 2.05, 2.35, 2.6]) / 2.6
    # plt.scatter(jay_06_r17x2, jay_06_r17y2)
    # energy3 = np.linspace(jay_06_r17x2[0], jay_06_r17x2[-1])
    # signal3 = get_signal(0, 6, "R", 1, 17, energy3, 1475)
    # plt.plot(energy3, signal3 / max(signal3))

    plt.xlabel("Laser Energy [J]")
    plt.ylabel("Signal")
    plt.show()


if __name__ == "__main__":
    main()
