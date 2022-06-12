"""
This script solves an aerodynamic blade and airfoil design optimization problem.

The L/D, CD, and stall margin values across the blade are controlled by the
optimizer to fine-tune the airfoil shapes to maximize AEP. Additionally,
the optimal twist profile across the blade is solved by using the INN
and can be considered an implicit design variable.
"""

import os

from wisdem import run_wisdem


mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = os.path.dirname(mydir) + os.sep + "IEA-15-240-RWT.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)