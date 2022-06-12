"""
This example script performs aerodynamic optimization of the IEA 15MW
blade by varying only the twist while keeping all airfoil designs fixed.

This is a simple blade design problem that does not include any variables
to control the airfoil shapes. We still use the INN to obtain the aerodynamic
performance of the fixed airfoil shapes. These shapes do not vary during the
optimization; only the twist. Here, we maximize AEP.
"""


import os

from wisdem import run_wisdem


mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = os.path.dirname(mydir) + os.sep + "IEA-15-240-RWT.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)