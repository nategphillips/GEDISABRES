# module state
'''
Holds the State object (for now).
'''

class State:
    def __init__(self, constants: list, vib_qn: int) -> None:
        self.elc_const = constants[0]
        self.vib_const = constants[1:5]
        self.rot_const = constants[5:12]
        self.spn_const = constants[12:14]
        self.vib_qn    = vib_qn

    def rotational_terms(self) -> list[float]:
        b_v = self.rot_const[0]                          - \
              self.rot_const[1] * (self.vib_qn + 0.5)    + \
              self.rot_const[2] * (self.vib_qn + 0.5)**2 + \
              self.rot_const[3] * (self.vib_qn + 0.5)**3

        d_v = self.rot_const[4] - self.rot_const[5] * (self.vib_qn + 0.5)

        h_v = self.rot_const[6]

        return [b_v, d_v, h_v]

    def electronic_term(self) -> float:
        return self.elc_const

    def vibrational_term(self) -> float:
        return self.vib_const[0] * (self.vib_qn + 0.5)    - \
               self.vib_const[1] * (self.vib_qn + 0.5)**2 + \
               self.vib_const[2] * (self.vib_qn + 0.5)**3 + \
               self.vib_const[3] * (self.vib_qn + 0.5)**4
