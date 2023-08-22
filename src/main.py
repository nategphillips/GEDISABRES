# module main
'''
Computes spectral lines for triplet oxygen.
'''

import numpy as np

import convolution as conv
import initialize as init
import constants as cn
import output as out
import input as inp
import energy
import state

# TODO: make an object for a vibrational band so that franck-condon factors will be easier
# TODO: only process the sample data if the user specified which comparisons they want

def normalization(intensity_data: list) -> np.ndarray:
    max_val = max(intensity_data[0])

    return np.array(intensity_data) / max_val

def main():
    # Temperature in Kelvin
    temp = 300
    # Pressure in Pa
    pres = 101325

    # Temperature used in Cosby is 300 K
    # Pressure used in Cosby is 20 Torr (2666.45 Pa)
    # pres = 2666.45

    # Range of desired rotational quantum numbers
    rot_qn_list = np.arange(0, 37, 1)

    # Instantiate the ground and excited states
    grnd_state = state.State(cn.X_CONSTS, 0) # X 3Sg-
    exct_state = state.State(cn.B_CONSTS, 2) # B 3Su-

    # Calculate the band origin energy
    band_origin = energy.get_band_origin(grnd_state, exct_state)
    #v_0 = 36185

    # Initialize the list of valid spectral lines
    lines = init.selection_rules(rot_qn_list)

    # Get the wavenumbers and intensities for each main branch (for plotting separately)
    branch_map = {'r': 0, 'p': 1, 'rq': 2, 'pq': 3}
    wns = [[line.wavenumber(band_origin, grnd_state, exct_state)
            for line in lines if line.branch == branch] for branch in branch_map]
    ins = [[line.intensity(band_origin, grnd_state, exct_state, temp)
            for line in lines if line.branch == branch] for branch in branch_map]
    ins = normalization(ins)

    # Combine the wavenumber and intensity data for use with the convolution function
    total_wn = [line.wavenumber(band_origin, grnd_state, exct_state) for line in lines]
    total_in = [line.intensity(band_origin, grnd_state, exct_state, temp) for line in lines]
    total_in = total_in / max(total_in)

    conv_data = None
    if inp.CONVOLVED_DATA:
        conv_data = conv.convolved_data(total_wn, total_in, temp, pres)

    # Fetch sample data for plotting
    smp_plot = out.sample_plotter()

    # TODO: fix without resorting to None
    out.plotter([wns, ins], conv_data, smp_plot)

if __name__ == '__main__':
    main()
