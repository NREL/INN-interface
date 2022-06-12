"""
This example simply runs the coupled INN-WISDEM code in an analysis, not an optimization.

This is the simplest type of case that uses the INN airfoil information within
the full turbine computation. Instead of using the tabulated airfoil performance
information contained in the geometry yaml, this example instead uses the performance
info computed by the INN.
"""

import os

from wisdem import run_wisdem
from wisdem.commonse.mpi_tools import MPI
from wisdem.postprocessing.compare_designs import run

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = os.path.dirname(mydir) + os.sep + "IEA-15-240-RWT.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)
