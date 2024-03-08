# todo

## code

- [x] 11/19/23 fix how the molecular constants are passed to the State class
  - ideally just add all of this data to a csv and then extract it with a pandas dataframe, which would make it a lot easier to add molecules in the future
- [ ] 11/20/23 try to automate the process of identifying potential vibrational overlaps for lif
    1) identify a range of wavenumbers / wavelengths to probe
    2) iterate over the quantized lines in the range for each vibrational band a set value are marked as potential transition candidates
    3) exclude satellite bands from the computation to reduce the number of steps
- [ ] 01/22/24 potentially implement separate partition functions
  - would allow rotational, vibrational, and electronic temperatures to be set separately
- [ ] 01/24/24 fix triplet line positions
  - see the note in energy.py about changing the sign of state.spn_consts[0]; the sign of the spin-rotation constant also needs to be double checked (it's a third-order term so it doesn't change too much -- still want to correctly apply it though)
  - also look at the Bergeman paper, it seems like he's using a more updated formula for calculating the triplet line positions (Herzberg uses the formulae proposed by Schlapp, which are from 1936)
  - when looking at my data vs. the HITRAN data, one of the three line positions for each triplet is misaligned
  - this problem is also seen when comparing against the PGOPHER data, but I think PGOPHER combines two of the triplet lines together since their wavenumber positions are too close to resolve without using higher-order corrections to the rotational term
- [ ] 01/24/24 add support for simulating any homonuclear diatomic molecules
  - allowing selection of the two state term symbols or Hund's case would change the selection rules automatically
- [ ] 01/24/24 properly implement the quantum numbers J and N (which is K in Herzberg)
  - I think this is one of the things causing issues with triplet splitting calculations
- [ ] 01/24/24 add types of convolution for the instrument function
  - only supports Gaussian currently

## docs

### General Theory

#### Molecular Approximations

- Rigid Rotator
  - Theory
  - Energy levels
  - Spectrum
- Harmonic Oscillator
- Anharmonic Oscillator
- Nonrigid Rotator
- Vibrating Rotator
- Symmetric Top

#### Structure of Electronic Transitions

- Total
- Electronic
- Vibrational
- Rotational
  - Branches

#### Intensities

- Vibrational
  - Partition function
  - Intensity formulae
  - Franck-Condon factors
- Rotational
  - Partition function
  - Intensity formulae
  - Hönl-London factors

#### Quantum Numbers

- Atoms
  - Single electron
  - Multi electron
  - Term symbols
- Molecules
  - Single electron
  - Multi electron
  - Term symbols

### Detailed Theory

#### Hund's Coupling Cases

- Case (a)
  - Good quantum numbers
  - Selection rules
  - Term values
- Case (b)
  - Good quantum numbers
  - Selection rules
  - Term values

#### Uncoupling

- Λ Doubling
- Spin uncoupling
- Term values

#### Electronic Transitions

- Σ-Σ
  - Branches
- Σ-Π
- Π-Σ
- Π-Π

### Individual Treatment

#### O2

#### O2+

#### NO

#### CO
