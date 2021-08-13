# VQA_CV_BoseHubbard

We provide here a sample python code for the simulationg of a Variational Quantum Algorithm (VQA) to encode a strongly correlated many-bosons wavefunction on a continuous-Variablale photnonic device. The targeted problem is here to encode ground state of the attractive Bose-Hubbard model. 

The code is based on two important sections:
    1. We build the matrix representation of the Hamiltonian and we exactly diagonalize it.
        This provides an exact reference for the ground state of the model.
    2. We simulate Variational Quantum Algorithms (VQA) by either minimizing the infidelity
       (or the energy) of the resulting trial state at the end of the circuit.

For the code to work, several packages are required:
- Strawberryfield (for the simulation of the CV photonic device), see: https://strawberryfields.ai/ 
- Optimparallel (a parallel version of the L-BFGS-B optimization algorithm), see https://pypi.org/project/optimparallel/
