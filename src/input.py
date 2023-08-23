# module input
'''
User-defined control inputs.
'''

# Lines
LINE_DATA: bool = True

# Convolution
CONVOLVED_DATA: bool = False

# Select which data to compare with
COMPARED_DATA: list[str] = ['pgopher', 'harvard']

# Plotting
PLOT_SAVE:  bool  = False
PLOT_PATH:  str   = '../img/example.webp'
DPI:        int   = 96
SCREEN_RES: tuple = (1920, 1080)
