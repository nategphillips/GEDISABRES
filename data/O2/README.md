# Sources for O2

## Einstein Coefficients

### B3Σu- - X3Σg-

(PRIMARY) ["Arrays of radiative transition probabilities for the N2 first and second positive, no beta and gamma, N+2 first negative, and O2 Schumann-Runge band systems" by Christophe O. Laux, Charles H. Kruger](https://doi.org/10.1016/0022-4073(92)90003-M)

["Absorption by vibrationally excited molecular oxygen in the Schumann-Runge continuum" by A.C. Allison, A. Dalgarno, N.W. Pasachoff](https://doi.org/10.1016/0032-0633(71)90007-9)

## Franck-Condon Factors

### B3Σu- - X3Σg-

(PRIMARY) ["The Potential Energy Curve for the B3∑−u State of Oxygen and Accurate Franck-Condon Factors for the Schumann-Runge Bands" by Cheung A.S.C., Mok D.K.W., Sun Y., Freeman D.E.](https://doi.org/10.1006/jmsp.1994.1002)

["Arrays of radiative transition probabilities for the N2 first and second positive, no beta and gamma, N+2 first negative, and O2 Schumann-Runge band systems" by Christophe O. Laux, Charles H. Kruger](https://doi.org/10.1016/0022-4073(92)90003-M)

## Per-level Constants

This program uses Herzberg's definition of the band origin, i.e. $\nu_0 = \nu_e + \nu_v$, which is given on p. 186 of "Spectra of Diatomic Molecules". In the Cheung paper, the electronic energy is defined differently than in Herzberg. The conversion specified by Cheung on p. 5 is $\nu_0 = T + 2 / 3 * \lambda - \gamma$. This conversion was applied to the raw data used in this program.

The Hamiltonians in Cheung and Yu are defined slightly differently, which leads to some constants having different values. Since the Cheung Hamiltonian matrix elements are used to solve for the energy eigenvalues, the constants from Yu are changed  to fit the convention used by Cheung. The Cheung convention also matches that of PGOPHER, which is my preferred convention (see <https://pgopher.chm.bris.ac.uk/Help/linham.htm>). The table below details the changes made to Yu's constants:

Cheung  | Yu
--------|------------
D       | -D
lamda_D | 2 * lamda_D
gamma_D | 2 * gamma_D

### X3Σg-

["High resolution spectral analysis of oxygen. IV. Energy levels, partition sums, band constants, RKR potentials, Franck-Condon factors involving the X3Σg-, a1Δg and b1Σg+ states" by Shanshan Yu, Brian J. Drouin, Charles E. Miller](https://doi.org/10.1063/1.4900510)

### B3Σu-

["Molecular spectroscopic constants of O2(B3Σu−): The upper state of the Schumann-Runge bands" by A.S.C. Cheung, K. Yoshino, W.H. Parkinson, D.E. Freeman](https://doi.org/10.1016/0022-2852(86)90196-7)

## Predissociation Factors

["Rotational variation of predissociation linewidth in the Schumann-Runge bands of 16O2" by B.R. Lewis, L. Berzins, J.H. Carver, S.T. Gibson](https://doi.org/10.1016/0022-4073(86)90068-3)
