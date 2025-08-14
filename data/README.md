# CSV Format and Constants

## Format of CSV Files Without Headers

If a file does not have a header indicating the data in the column, the format is as follows:

- Rows: Upper state vibrational quantum number (v')
- Columns: Lower state vibrational quantum number (v'')

## Constants

Sign convention for Dunham parameters is taken from Herzberg and Babou
(https://doi.org/10.1007/s10765-007-0288-6). Some papers report equilibrium constants for
non-standard terms like A, p, and q. These terms are treated with the same sign convention used for
B(v). Centrifugal distortion terms like A_D, p_D, and q_D follow the same sign convention used for
D(v). The sign convention for corrections to higher-order terms like H (i.e., α_{H_e} or γ_{H_e})
are not well-defined. In case they are ever provided, the sign convention is the same as that of
B(v).

In short, the second term in the expansion is subtracted except in the case of centrifugal
distortion.

Vibrational term:
$$
G(v) = ω_e(v + 0.5) − ω_ex_e(v + 0.5)^2 + ω_ey_e(v + 0.5)^3 + ...
$$

Rotational term:
$$
B(v) = B_e - α_e(v + 0.5) + γ_e(v + 0.5)^2 + ...
$$

Coupling terms:
$$
A(v) = A_e - α_{A_e}(v + 0.5) + γ_{A_e}(v + 0.5)^2 + ...
p(v) = p_e - α_{p_e}(v + 0.5) + γ_{p_e}(v + 0.5)^2 + ...
$$

Centrifugal distortion terms:
$$
D(v)   = D_e     + β_e(v + 0.5)         + g_e(v + 0.5)^2         + ...
A_D(v) = A_{D_e} + α_{A_{D_e}}(v + 0.5) + γ_{A_{D_e}}(v + 0.5)^2 + ...
$$

Higher-order terms:
$$
H(v)   = H_e     + α_{H_e}(v + 0.5)     + γ_{H_e}(v + 0.5)^2     + ...
A_H(v) = A_{H_e} + α_{A_{H_e}}(v + 0.5) + γ_{A_{H_e}}(v + 0.5)^2 + ...
$$

The constants in the data files should be input exactly as they appear in the papers from which
they were obtained, assuming the papers follow the sign convention of Herzberg. Unfortunately, not
all papers use this convention, meaning one must be careful to ensure that the definitions in the
paper match those given here.

Often times, the data from papers will report a different electronic energy value T_e than the NIST
values tabulated in constants.py. For this reason, the first column in each Dunham constants file
reports T_e on the first row.
