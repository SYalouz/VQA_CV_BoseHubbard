"""
This is an example of source code used to study the performance of a photonic based quantum ansatz 
develpped to encode strongly correlated many-boson wavefunctions. The stargeted problem 
is here to encode the groundstate of the attractive Bose-Hubbard Hamiltonian.

The code is based on two important ingredients:
    1. We build the matrix representation of the Hamiltonian and we exactly diagonalize it.
        This provides an exact reference for the ground state of the model.
    2. We simulate Variational Quantum Algorithms (VQA) by either minimizing the infidelity
       (or the energy) of the resulting trial state at the end of the circuit.
"""

import numpy as np
import strawberryfields as sf
from strawberryfields.ops import *
import math as m
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy.optimize import minimize
from optimparallel import minimize_parallel
from itertools import combinations_with_replacement, product
from scipy import linalg
from random import random

def Cost_function_Energy(param_values,
                         param_names,
                         CV_circuit,
                         H,
                         list_focktates,
                         N_S):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Cost function to compute the energy of a trial state 
    at the end of the quantum circuit 
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """
    params_dict = ({})
    for name, value in zip(param_names, param_values):
        params_dict.update({ name : value })
    
    circuit_results = eng.run( CV_circuit, args = params_dict )
    ket = circuit_results.state.ket() 
    Energy = 0
    for kappa in range(len(H)):
        occ_kappa = list_focktates[kappa]
        index = tuple( occ_kappa[i] for i in range(N_S) )
        for kappa_ in range(len(H)):
            occ_kappa_ = list_focktates[kappa_]
            index_ = tuple( occ_kappa_[i] for i in range(N_S) )
            Energy += (np.conj( ket[index_]) * H[ kappa_, kappa ] * ket[index]  )
    # print(Energy.real)
    return Energy.real 
        
def Cost_function_infidelity(param_values,
                             param_names,
                             CV_circuit,
                             H,
                             list_focktates,
                             EXACT_GS,
                             N_S):
    """
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    Cost function to compute the infidelity of a trial state 
    at the end of the quantum circuit 
    >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    """
    params_dict = ({})
    for name, value in zip(param_names, param_values):
        params_dict.update({ name : value })
     
    circuit_results = eng.run( CV_circuit, args = params_dict )
    ket = circuit_results.state.ket() 
    
    Overlap = 0.
    for kappa in range(len(H)):
        occ_kappa = list_focktates[kappa] 
        index = tuple( occ_kappa[i] for i in range(N_S) )
        Overlap += np.conj( ket[ index ] ) * EXACT_GS[kappa]
    # print( "Infidelity (%)", (1 - abs(Overlap)**2.)*100. )
    return 1 - abs(Overlap)**2.

# ====================================================
# Parameters for the simulation 
MAX_ITER_OPT = 20000 # Maximum number of iteration for the VQAs
N_layer      = 8     # Number of layers considered in the ansatz circuit
dis_amp      = 0.1   # Disroder amplitude for the initialization of the circuits parameters
N_B          = 8     # Total number of bosons
N_S          = 2     # Total number of sites
LAMBDA       = 0.01  # Many-body interaction term
J            = 1.    # Hopping amplitude (reduced unit of energy here)
U            = LAMBDA * J / N_B # Local many body interaction

# ====================================================
# === Exact diagonalization of the BH Hamiltonian ====

# Total dimension of the Hamiltonian
dim_H = m.factorial(N_B + N_S - 1) // ( m.factorial(N_B)*m.factorial(N_S-1) )

# Definition of the Hopping matrix ( connexions between the sites of a BH network )
Hoppings = np.zeros(( N_S, N_S ))
for site in range(N_S-1):
    Hoppings[site,site+1] = Hoppings[site+1,site] = -J 
Hoppings[0,-1] = Hoppings[-1,0] = -J

# Definition of the local attractive interactions
U_site  = np.zeros( N_S )
for site in range(N_S):
    U_site[ site ]  = U

list_focktates = []
for combination in combinations_with_replacement( range(N_S), N_B ):
    fockstate = [ 0 for i in range(N_S) ]
    for index in list(combination):
        fockstate[ index ] += 1
    list_focktates += [ fockstate ]

# Building each element of the attractive BH hamiltonian matrix
H = np.zeros(( dim_H, dim_H ))
for kappa in range(dim_H):
    ref_fockstate = list_focktates[ kappa ]
    
    for site in range(N_S):
        N_B_site = ref_fockstate[ site ]
        
        if ( N_B_site > 0 ):
            H[ kappa, kappa ] += -0.5 * U_site[ site ] * N_B_site * ( N_B_site - 1. ) 
            if ( site < N_S ):
                for new_site in range(site+1,N_S):
                    N_B_new_site  = ref_fockstate[ new_site ]
                    new_fockstate = ref_fockstate.copy()
                    new_fockstate[site]     += -1
                    new_fockstate[new_site] += 1
                    kappa_ = list_focktates.index( new_fockstate )
                    H[ kappa_, kappa ] =  H[ kappa, kappa_ ]  = ( Hoppings[new_site,site] * m.sqrt( ( N_B_new_site + 1 ) * N_B_site )  )         

eigen_energies, eigen_states = linalg.eigh( H )    
EXACT_GS = eigen_states[:,0]
EXACT_GS_ENERGY = eigen_energies[0]


# =====================================
# ========   VQA simulation   =========

# Building the CV quantum circuit with the initial state already encoded
CV_circuit = sf.Program( N_S )
cutoff = N_B + 1
eng  = sf.Engine("fock", backend_options={'cutoff_dim': cutoff})
sf.hbar = 1
param_names  = []
param_values = [] 
ket_initial = np.zeros( [cutoff]*N_S, dtype=np.complex64 )

# Building a monomodal initial state
num_local_bosons = [ N_B ] 
for i in range(N_S-1):
    num_local_bosons += [ 0 ]
ket_initial[ tuple(num_local_bosons) ] = 1.0 + 0.0j

# Building the minimal BS-Kerr ansatz
with CV_circuit.context as q:
    sf.ops.Ket(ket_initial) | q
    gate_ind  = 0
    prev_site = 0
    
    for layer in range(N_layer): 
        # We build here a given layer:
        for n_mode in range(N_S-1): # First a stair of Beam-splitter
            if (layer%2 == 0):
                param_values += [ dis_amp * (random()-0.5)  ]
                param_names  += [ CV_circuit.params('theta_{}'.format(gate_ind)) ]
                BSgate( param_names[gate_ind], 0) | (q[n_mode], q[n_mode+1] )
                gate_ind += 1  
            else:
                param_values += [ dis_amp * (random()-0.5)  ]
                param_names  += [ CV_circuit.params('theta_{}'.format(gate_ind)) ]
                BSgate( param_names[gate_ind], 0) | (q[N_S-1-n_mode], q[N_S-1-(n_mode+1)] )
                gate_ind += 1  

        for n_mode in range(N_S): # Followed by a series of Kerr gate for each mode
            param_values += [ dis_amp * (random()-0.5) ] 
            param_names  += [ CV_circuit.params('theta_{}'.format(gate_ind)) ]
            Kgate( param_names[gate_ind]) | (q[n_mode] ) 
            gate_ind += 1                                         

# VQA to minimize the infidelity of a trial state
f_min = minimize_parallel( Cost_function_infidelity,
                              x0      = (param_values),
                              args    = (param_names,
                                          CV_circuit,
                                          H,
                                          list_focktates,
                                          EXACT_GS,
                                          N_S),
                              options = {'maxiter': MAX_ITER_OPT} )

# # VQA to minimize the energy of a trial state (i.e. VQE) 
# f_min = minimize( Cost_function_Energy,
#                 x0      = (param_values),
#                 args    = (param_names,
#                             CV_circuit,
#                             H,
#                             list_focktates,
#                             N_S),
#                 method  = 'BFGS',
#                 options = {'maxiter': MAX_ITER_OPT})

# Building the optimized quantum state obtained with the ansatz
param_values = f_min['x']
params_dict  = ({})
for name, value in zip(param_names, param_values):
    params_dict.update({ name : value })
circuit_results = eng.run( CV_circuit, args = params_dict )
ket_final = circuit_results.state.ket() 

# Computing the final fidelity and energy of the quantum state
Infidelity = Cost_function_infidelity(param_values,
                                     param_names,
                                     CV_circuit,
                                     H,
                                     list_focktates,
                                     EXACT_GS,
                                     N_S)
Fidelity = (1. - Infidelity) * 100.

Energy = Cost_function_Energy(param_values,
                                 param_names,
                                 CV_circuit,
                                 H,
                                 list_focktates,
                                 N_S)
    
print("Fidelity ( % )        : ", Fidelity )
print("Exact GS energy       : ", EXACT_GS_ENERGY)
print("Ansatz state's energy : ", Energy)
print("|Energy error|        : ", abs(EXACT_GS_ENERGY - Energy))