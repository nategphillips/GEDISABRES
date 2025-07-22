# Notes on Data for O2

References to Cheung refer to "Molecular spectroscopic constants of O2(B3Σu−): The upper state of the Schumann-Runge bands" by Cheung, et al.

References to Yu refer to "High resolution spectral analysis of oxygen. IV. Energy levels, partition sums, band constants, RKR potentials, Franck-Condon factors involving the X3Σg-, a1Δg, and b1Σg+ states" by Yu et al.

This program uses Herzberg's definition of the band origin, i.e. nu_0 = nu_e + nu_v, which is given on p. 186 of "Spectra of Diatomic Molecules". In the Cheung paper, the electronic energy is defined differently than in Herzberg. The conversion specified by Cheung on p. 5 is nu_0 = T + 2 / 3 * lamda - gamma. This conversion was applied to the raw data used in this program.

The Hamiltonians in Cheung and Yu are defined slightly differently, which leads to some constants having different values. Since the Cheung Hamiltonian matrix elements are used to solve for the energy eigenvalues, the constants from Yu are changed  to fit the convention used by Cheung. The Cheung convention also matches that of PGOPHER, which is my preferred convention (see https://pgopher.chm.bris.ac.uk/Help/linham.htm). The table below details the changes made to Yu's constants:

Cheung  | Yu
--------|------------
D       | -D
lamda_D | 2 * lamda_D
gamma_D | 2 * gamma_D
