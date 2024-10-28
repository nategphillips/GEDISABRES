# module lif

import matplotlib.pyplot as plt
import numpy as np
import scipy as sy

import constants as cn


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


def main():
    # Testing parameters for the (15, 3) R1(11) transition.
    g_l: int = 3
    g_u: int = 1
    j_qn: int = 11
    f_b: float = 0.127
    s_j: float = 12.96
    nu: float = 56450  # [1/cm]
    nu_d: float = 0.25  # [1/cm]
    w_d: float = 2 * np.pi * cn.LIGHT * nu_d  # [1/s]
    a_21: float = 2.658e6 * s_j / (2 * j_qn + 1)  # [1/s]
    w_f: float = a_21 * 10  # [1/s]
    b_12: float = a_21 / (8 * np.pi * cn.PLANC * cn.LIGHT * nu**3) * g_u / g_l  # [cm/J]
    b_21: float = b_12 * g_l / g_u  # [cm/J]

    pressure: float = 1.0  # [atm]
    temperature: float = 300.0  # [K]
    pulse_center: float = 30e-9  # [s]
    pulse_width: float = 20e-9  # [s]
    laser_energy: float = 25e-3  # [J]
    laser_area: float = 1.0  # [cm^2]
    fluence: float = laser_energy / laser_area

    w_c: float = 7.78e9 * pressure * np.sqrt(300 / temperature)  # [1/s]
    w_q: float = 7.8e9 * pressure * np.sqrt(300 / temperature)  # [1/s]

    print(f"w_c: {w_c:.4e}")
    print(f"w_d: {w_d:.4e}")
    print(f"w_f: {w_f:.4e}")
    print(f"w_q: {w_q:.4e}")
    print(f"a_21: {a_21:.4e}")
    print(f"b_12: {b_12:.4e}")
    print(f"b_21: {b_21:.4e}")

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
    s_f /= n2.max()

    i_l = laser_intensity(t, pulse_center, pulse_width, fluence)
    i_l /= i_l.max()

    _, ax1 = plt.subplots()

    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("N1, N3, IL")
    ax1.plot(t, n1, label="N1")
    ax1.plot(t, n3, label="N3")
    ax1.plot(t, i_l, label="IL")

    ax2 = ax1.twinx()
    ax2.set_ylabel("N2, SF")
    ax2.plot(t, n2, label="N2")
    ax2.plot(t, s_f, label="SF")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
