## Installing the INN-interface
Installation with [Anaconda](https://www.anaconda.com) is the recommended approach because of the ability to create self-contained environments suitable for testing and analysis.  The WISDEM-INN framework requires [Anaconda 64-bit](https://www.anaconda.com/distribution/).

The Anaconda environment can be built from the environment.yml using

        conda env create -f environment.yml
        conda activate inn_env

Then install the TensorFlow using

        pip install 'tensorflow>=2.7.0'

Then install the INN-interface library

        git clone https://github.com/NREL/INN-interface.git
        cd INN-interface/
        pip install -e .
        cd ..

## Installation with WISDEM®
To install the INN-interface with WISDEM follow these steps.

1.  In order to directly use the examples in the repository and peek at the code when necessary, we recommend all users install WISDEM in *developer* mode. This can be accomplished by following the installation steps on the [WISDEM](https://github.com/WISDEM/WISDEM) page. Alternatively, we can simply instal WISDEM as a conda package to easily satisfy all dependencies. Then, we remove the WISDEM conda package and reinstal from the Github source code. Note the differences between Windows and Mac/Linux build systems:

        conda install -y wisdem
        conda remove --force wisdem
        conda install -y compilers mpi4py petsc4py   # (Mac / Linux only)
        conda install -y m2w64-toolchain libpython   # (Windows only)
        pip install marmot-agents
        git clone https://github.com/WISDEM/WISDEM.git
        cd WISDEM
        python setup.py develop
        cd ..

2. OPTIONAL: Install pyOptSparse, a package that provides a handful of additional optimization solvers and has OpenMDAO support:

        git clone https://github.com/evan-gaertner/pyoptsparse.git
        cd pyoptsparse
        python setup.py install
        cd ..

3. Try running an example

        cd INN-interface/INN_interface/examples/01_run_WISDEM
        python blade_driver.py 
 
