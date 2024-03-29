import os

from wisdem import run_wisdem
from wisdem.commonse.mpi_tools import MPI
from wisdem.postprocessing.compare_designs import run

mydir = os.path.dirname(os.path.realpath(__file__))  # get path to this file
fname_wt_input = os.path.dirname(mydir) + os.sep + "IEA-15-240-RWT.yaml"
fname_modeling_options = mydir + os.sep + "modeling_options.yaml"
fname_analysis_options = mydir + os.sep + "analysis_options.yaml"

wt_opt, modeling_options, analysis_options = run_wisdem(fname_wt_input, fname_modeling_options, fname_analysis_options)

# cd = wt_opt['blade.interp_airfoils.cd_interp'][:, 110, 0, 0]
# cl = wt_opt['blade.interp_airfoils.cl_interp'][:, 110, 0, 0]
# r_thick = wt_opt['blade.interp_airfoils.r_thick_interp']
# 
# 
# print('cd', repr(cd))
# print('cl', repr(cl))
# print('L/D', repr(cl / cd))
# print('t/c', repr(r_thick))


# if MPI:
#     rank = MPI.COMM_WORLD.Get_rank()
# else:
#     rank = 0
# 
# if rank == 0:
#     print(
#         "RUN COMPLETED. RESULTS ARE AVAILABLE HERE: "
#         + os.path.join(mydir, analysis_options["general"]["folder_output"])
#     )
# 
# run([wt_opt], ["optimized"], modeling_options, analysis_options)
