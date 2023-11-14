#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import bisect
import argparse

_irreps_ = np.zeros(shape=(1), dtype=int)
_epsilon_ = np.zeros(shape=(1))
_G_ = np.zeros(shape=(1,1))
_matrices_ = [np.zeros(shape=(1,1))]
_N_ = 1
_Np_ = 14
_type_ = float
_mu_ = 0
_V0_ = 1

def set_global_variables(irreps, epsilon, G, COMPLEX = False):
    global _irreps_, _epsilon_, _G_, _matrices_, _N_, _type_
    _irreps_ = irreps
    _epsilon_ = epsilon
    _G_ = G
    _N_ = len(irreps)
    if COMPLEX:
        _type_ = complex
    else:
        _type_ = float
    _matrices_ = []
    for n in range(_N_):
        matrix = np.zeros(shape=(_irreps_[n],_irreps_[n]),dtype=_type_)
        for i in range(0,_irreps_[n]//2):
            matrix[i, i] = -_epsilon_[n]
            matrix[_irreps_[n]-1-i, _irreps_[n]-1-i] = _epsilon_[n]
        _matrices_.append(matrix)


def update_matrices(Delta):
    for n in range(_N_):
        for i in range(0, _irreps_[n]//2):
            _matrices_[n][i,_irreps_[n]-1-i] = -np.conj(Delta[n])
            _matrices_[n][_irreps_[n]-1-i,i] = -Delta[n]


def diagonalize_and_update_Delta(Delta_in):
    eigenvalues = [np.zeros(shape=(1))] * _N_
    eigenvectors = [np.zeros(shape=(1,1))] * _N_
    for n in range(_N_):
        eigenvalues[n], eigenvectors[n] = np.linalg.eigh(_matrices_[n])
    Delta_out = np.zeros(shape=(_N_),dtype=_type_)
    for n1 in range(_N_):
        for n2 in range(_N_):
            for i in range(_irreps_[n2]//2):
                Delta_out[n1] += _G_[n1, n2] * np.vdot(eigenvectors[n2][i,0:_irreps_[n2]//2], eigenvectors[n2][_irreps_[n2]-1-i,0:_irreps_[n2]//2])
    return Delta_out


def self_consistent_diagonalization(Delta_initial, threshold_error=1e-12, max_iterations=100, min_iterations=1, PRINT=False):
    Delta_old = np.zeros(shape=(_N_))
    Delta_new = Delta_initial
    current_error = np.linalg.norm(Delta_new)
    iteration = 1
    while current_error > threshold_error and iteration <= max_iterations or iteration <= min_iterations:
        update_matrices(Delta_new)
        Delta_old = Delta_new
        Delta_new = diagonalize_and_update_Delta(Delta_old)
        if PRINT:
            print_data(Delta_new, 'Δ', "\nIteration %i -- Current error: %e\n" % (iteration,current_error))
        current_error = np.linalg.norm(Delta_new - Delta_old)
        iteration = iteration + 1
    return Delta_new


def self_consistent_gap_equation(Delta_initial, threshold_error=1e-12, max_iterations=100, min_iterations=1, PRINT=False):
    Delta_old = np.zeros(shape=(_N_), dtype=_type_)
    Delta_new = Delta_initial
    current_error = np.linalg.norm(Delta_new)
    iteration = 1
    while current_error > threshold_error and iteration <= max_iterations or iteration <= min_iterations:
        Delta_old = np.copy(Delta_new)
        Summand_Gap_Equation = np.array([0.25*_irreps_[n] * Delta_old[n]/np.sqrt(_epsilon_[n]**2 + Delta_old[n]**2) for n in range(_N_)])
        for n in range(_N_):
            Delta_new[n] = np.dot(_G_[n,:],Summand_Gap_Equation)
        if PRINT:
            print_data(Delta_new, 'Δ', "\nIteration %i -- Current error: %e\n" % (iteration,current_error))
        current_error = np.linalg.norm(Delta_new - Delta_old)
        iteration = iteration + 1
    update_matrices(Delta)
    return Delta_new


def calculate_occupation(Delta):
    occupation = np.zeros(shape=(_N_))
    eigenvalues = [np.zeros(shape=(1))] * _N_
    eigenvectors = [np.zeros(shape=(1,1))] * _N_
    for n in range(_N_):
        eigenvalues[n], eigenvectors[n] = np.linalg.eigh(_matrices_[n])
    for n in range(_N_):
        for i in range(_irreps_[n]//2):
            occupation[n] += np.vdot(eigenvectors[n][_irreps_[n]-1-i,0:_irreps_[n]//2], eigenvectors[n][_irreps_[n]-1-i,0:_irreps_[n]//2])
            occupation[n] -= np.vdot(eigenvectors[n][i,0:_irreps_[n]//2], eigenvectors[n][i,0:_irreps_[n]//2])
        occupation[n] += _irreps_[n]//2
    return occupation


def calculate_energy(Delta):
    energy = 0.
    eigenvalues = [np.zeros(shape=(1))] * _N_
    eigenvectors = [np.zeros(shape=(1,1))] * _N_
    for n in range(_N_):
        eigenvalues[n], eigenvectors[n] = np.linalg.eigh(_matrices_[n])
        for i in range(len(eigenvalues[n])):
            if eigenvalues[n][i] < 0:
                energy += eigenvalues[n][i]
        energy +=  0.5* _irreps_[n] * _epsilon_[n]
        for i in range(_irreps_[n]//2):
            energy += Delta[n] * np.vdot(eigenvectors[n][_irreps_[n]-1-i,0:_irreps_[n]//2], eigenvectors[n][i,0:_irreps_[n]//2])
    return energy


def check_if_eigenvector(matrix, eigenvector, eigenvalue):
    vector = np.dot(matrix, eigenvector)
    for i in range(len(vector)):
        if abs(vector[i] - eigenvalue*eigenvector[i]) > 1e-8:
            return False
    return True


def print_data(data, symbol, label=""):
    print(label, end='')
    for n in range(_N_):
        if _type_ == complex and np.iscomplexobj(data[0]):
            print(f"{symbol}[{_irreps_[n]-1}/2] = %f+%fi " % (data[n].real, data[n].imag),end='\t')
        else:
            print(f"{symbol}[{_irreps_[n]-1}/2] = %f " % data[n],end='\t')
    print("")


def check_epsilon_and_G(irreps, epsilon, G, PRINT=False):
    for n in range(len(irreps)):
        if PRINT:
            pass
            print(f"ε[{irreps[n]-1}/2] = {epsilon[n]+_mu_}")
    for n1 in range(len(irreps)):
        for n2 in range(n1, len(irreps)):
            if PRINT:
                print(f"G[{irreps[n1]-1}/2,{irreps[n2]-1}/2] = {G[n1,n2]}")
            assert abs(G[n1,n2] - G[n2,n1]) < 1e-10

            
def get_parameters_for_Sn(mu=0):
    global _mu_
    _mu_ = mu
    irreps = np.zeros(shape=(5), dtype=int)
    epsilon = np.zeros(shape=(5))
    V = np.zeros(shape=(5,5))
    G = np.zeros(shape=(5,5))
    irreps[0] = 8
    irreps[1] = 6
    irreps[2] = 4
    irreps[3] = 2
    irreps[4] = 12
    epsilon[0] = -6.121 - mu
    epsilon[1] = -5.508 - mu
    epsilon[2] = -3.749 - mu
    epsilon[3] = -3.891 - mu
    epsilon[4] = -3.778 - mu
    V[0,0] = 0.9850
    V[0,1] = 0.5711
    V[0,2] = 0.5184
    V[0,3] = 0.2920
    V[0,4] = 1.1454
    V[1,0] = 0.5711
    V[1,1] = 0.7063
    V[1,2] = 0.9056
    V[1,3] = 0.3456
    V[1,4] = 0.9546
    V[2,0] = 0.5184
    V[2,1] = 0.9056
    V[2,2] = 0.4063
    V[2,3] = 0.3515
    V[2,4] = 0.6102
    V[3,0] = 0.2920
    V[3,1] = 0.3456
    V[3,2] = 0.3515
    V[3,3] = 0.7244
    V[3,4] = 0.4265
    V[4,0] = 1.1454
    V[4,1] = 0.9546
    V[4,2] = 0.6102
    V[4,3] = 0.4265
    V[4,4] = 1.0599
    
    for i in range(5):
        for j in range(5):
            G[i,j] = V[i,j] * _V0_ * 2. / np.sqrt(irreps[i]*irreps[j])
    return irreps, epsilon, G

def run_fixed_mu(mu, max_iterations=1000, PRINT=True):
	
	irreps, epsilon, G = get_parameters_for_Sn(mu=mu)
	set_global_variables(irreps, epsilon, G, COMPLEX=False)
	#check_epsilon_and_G(irreps, epsilon, G, PRINT=True)
	
	Delta = np.array([1.0] * len(irreps))
	Delta = self_consistent_diagonalization(Delta, max_iterations=max_iterations, PRINT=PRINT)
	#Delta = self_consistent_gap_equation(Delta, max_iterations, PRINT)
	
	energy = calculate_energy(Delta)
	occupation = calculate_occupation(Delta)
	density = np.array([occupation[n]/_irreps_[n] for n in range(_N_)])
	
	return Delta, energy, occupation, density

def f_mu(x):
	
	global _Np_
	Delta, energy, occupation, density = run_fixed_mu(x, 100, False)
	return _Np_-sum(occupation)

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-save_Nmu', action='store_true', default=False)
	parser.add_argument('-V0', type=float, default=1)
	parser.add_argument('-Np', type=int, default=14)
	args = parser.parse_args()
	
	_V0_ = args.V0
	_Np_ = args.Np
	
	mu_opt = bisect(f_mu, -10., 0.)
	Delta, energy, occupation, density = run_fixed_mu(mu_opt, 1000, True)
	
	print("\nResult of self-consistent groundstate calculation:\n")
	print(f"Ground state energy:  \tE_0 = {energy + _mu_*sum(occupation)}")
	print(f"chemical potential:  \tμ = {mu_opt}")
	print(f"Total particle number:\tN = {sum(occupation)}")
	print_data(Delta, 'Δ', "Pairing gaps: \t")
	print_data(occupation, 'N', "Occupation: \t\t")
	print_data(density, 'n', "Density: \t\t")
	
	if args.save_Nmu:
		mu_vals = np.linspace(-10.,0.,501,endpoint=True)
		N_vals = []
		for mu_val in mu_vals:
			Delta, energy, occupation, density = run_fixed_mu(mu_val,1000,False)
			N_vals.append(sum(occupation))
		np.savetxt("N_mu_V0="+str(_V0_)+".dat", np.array([mu_vals, N_vals]).T)
    