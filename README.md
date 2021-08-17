## Installation with WISDEM
To install the INN-interface with WISDEM follow these steps.
Installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  The WISDEM-INN framework requires [Anaconda 64-bit](https://www.anaconda.com/distribution/).

The installation instructions below use the environment name, "wisdem_inn-env," but any name is acceptable.

1.  Setup and activate the Anaconda environment from a prompt (Anaconda3 Power Shell on Windows or Terminal.app on Mac)

        conda config --add channels conda-forge
        conda create -y --name wisdem_inn-env python=3.8
        conda activate wisdem_inn-env

2.  In order to directly use the examples in the repository and peek at the code when necessary, we recommend all users install WISDEM in *developer* mode.  This is done by first installing WISDEM as a conda package to easily satisfy all dependencies, but then removing the WISDEM conda package and reinstalling from the Github source code.  Note the differences between Windows and Mac/Linux build systems:

        conda install -y wisdem git
        conda remove --force wisdem
        conda install compilers mpi4py petsc4py   # (Mac / Linux only)
        conda install m2w64-toolchain libpython   # (Windows only)
        pip install simpy marmot-agents nlopt
        git clone https://github.com/WISDEM/WISDEM.git
        cd WISDEM
        git checkout af_design
        python setup.py develop


3. OPTIONAL: Install pyOptSparse, a package that provides a handful of additional optimization solvers and has OpenMDAO support:

        cd ..
        git clone https://github.com/evan-gaertner/pyoptsparse.git
        cd pyoptsparse
        python setup.py install
        cd ..

4. Install the INN-interface library

        pip install tensorflow==2.2 psdr
        git clone https://github.com/NREL/INN-interface.git
        cd INN-interface/
        pip install -e .

5. Try running an example

        cd INN_interface/examples/01_run_WISDEM
        python blade_driver.py 
 