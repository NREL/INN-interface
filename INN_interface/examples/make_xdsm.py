from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC


x = XDSM()

# Instantiate on-diagonal blocks
# order of args goes: object name, type of object, string for block on XDSM
x.add_system('opt', OPT, r'\text{Optimizer}')
x.add_system('inverse', FUNC, r'\text{Inverse design}')
x.add_system('polars', FUNC, r'\text{Generate polars}')
x.add_system('rp', FUNC, r'\text{Calculate rotor metrics}')

# Feed-forward connections; from, to, name for connections
x.connect('opt', 'inverse', r'Re, C_D, L/D, \text{stall margin}, t/c')
x.connect('inverse', 'polars', r'CST, \alpha_{inn}')
x.connect('polars', 'rp', r'\text{Lift and drag polars}')
x.connect('inverse', 'rp', r'\text{Airfoil shapes}')
x.connect('opt', 'rp', r'\text{Twist and chord profiles}')

# Feed-backward connections
x.connect('rp', 'opt', r'\text{Performance, loads, masses}')

# Outputs on the left-hand side
# x.add_output('opt', 'x^*, z^*', side='left')

x.add_process(["opt", "inverse", "polars", "rp", "opt"])

# Compile latex and write pdf
x.write('xdsm_inn_wisdem')