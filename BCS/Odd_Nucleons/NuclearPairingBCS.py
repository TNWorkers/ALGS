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
_Np_ = 13
_type_ = float
_mu_ = 0
_V0_ = 1
_EL_ = 'Sn'
_L_ = 8


def self_consistent_diagonalization(Delta_initial = [], threshold_error=1e-12, max_iterations=100, min_iterations=1, PRINT=False):
    Delta_old = np.zeros(shape=(_N_))
    if len(Delta_initial) != _N_:
        Delta_new = np.array([1.0] * _N_)
    else:
        Delta_new = Delta_initial
    current_error = np.linalg.norm(Delta_new)
    iteration = 1
    while current_error > threshold_error and iteration <= max_iterations or iteration <= min_iterations:
        update_matrices_offdiagonal(Delta_new)
        Delta_old = Delta_new
        Delta_new = diagonalize_and_calculate_Delta(Delta_old)
        if PRINT and False:
            print_data(Delta_new, 'Δ', "\nIteration %i -- Current error: %e\n" % (iteration,current_error))
        current_error = np.linalg.norm(Delta_new - Delta_old)
        iteration = iteration + 1
    update_matrices_offdiagonal(Delta_new)
    return Delta_new


def diagonalize_and_calculate_Delta(Delta_in):
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


def update_matrices_offdiagonal(Delta):
    global _matrices_
    for n in range(_N_):
        for i in range(0, _irreps_[n]//2):
            _matrices_[n][i,_irreps_[n]-1-i] = -np.conj(Delta[n])
            _matrices_[n][_irreps_[n]-1-i,i] = -Delta[n]


def update_matrices_diagonal(mu):
    global _mu_, _matrices_
    _mu_ = mu
    for n in range(_N_):
        for i in range(0,_irreps_[n]//2):
            _matrices_[n][i, i] = -_epsilon_[n] + _mu_
            _matrices_[n][_irreps_[n]-1-i, _irreps_[n]-1-i] = _epsilon_[n] - _mu_


def set_global_variables(irreps, epsilon, G, mu, COMPLEX = False):
    global _irreps_, _epsilon_, _G_, _mu_, _matrices_, _N_, _type_
    _irreps_ = irreps.copy()
    _epsilon_ = epsilon
    _G_ = G
    _N_ = len(irreps)
    _mu_ = mu
    if COMPLEX:
        _type_ = complex
    else:
        _type_ = float
    _matrices_ = [np.zeros(shape=(_irreps_[n],_irreps_[n]),dtype=_type_) for n in range(_N_)]
    update_matrices_diagonal(_mu_)


def initialize(mu=0, irrep_to_decrement=-1):
    if _EL_ == 'Sn':
        irreps, epsilon, G = get_parameters_for_Sn(irrep_to_decrement)
    elif _EL_ == 'Pb':
        irreps, epsilon, G = get_parameters_for_Pb(irrep_to_decrement)
    elif _EL_ == 'Richardson':
        irreps, epsilon, G = get_parameters_for_Richardson(irrep_to_decrement)
    else:
        irreps, epsilon, G = get_test_parameters()
    set_global_variables(irreps, epsilon, G, mu, COMPLEX=False)


def calculate_energy():
    energy = 0.
    eigenvalues = np.array([np.zeros(shape=(_irreps_[n])) for n in range(_N_)], dtype=np.ndarray)
    eigenvectors = np.array([np.zeros(shape=(_irreps_[n], _irreps_[n])) for n in range(_N_)], dtype=np.ndarray)
    for n in range(_N_):
        eigenvalues[n], eigenvectors[n] = np.linalg.eigh(_matrices_[n])
    for n in range(_N_):
        for i in range(_irreps_[n]//2):
            energy += _epsilon_[n] * np.vdot(eigenvectors[n][_irreps_[n]-1-i,:_irreps_[n]//2], eigenvectors[n][_irreps_[n]-1-i,:_irreps_[n]//2]).real
            energy -= _epsilon_[n] * np.vdot(eigenvectors[n][i,:_irreps_[n]//2], eigenvectors[n][i,:_irreps_[n]//2]).real
            energy += _epsilon_[n]
    coefficients = np.zeros(shape=(_N_))
    for n in range(_N_):
        for i in range(_irreps_[n]//2):
            coefficients[n] += np.vdot(eigenvectors[n][_irreps_[n]-1-i,:_irreps_[n]//2], eigenvectors[n][i,:_irreps_[n]//2]).real
    for n in range(_N_):
        for m in range(_N_):
            energy -= _G_[n,m] * coefficients[n] * coefficients[m]
    return energy


def calculate_occupation():
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



def check_if_eigenvector(matrix, eigenvector, eigenvalue):
    vector = np.dot(matrix, eigenvector)
    for i in range(len(vector)):
        if abs(vector[i] - eigenvalue*eigenvector[i]) > 1e-8:
            return False
    return True


def check_epsilon_and_G(irreps, epsilon, G, PRINT=False):
    for n in range(len(irreps)):
        if PRINT:
            pass
            print(f"ε[{irreps[n]-1}/2] = {epsilon[n]}")
    for n1 in range(len(irreps)):
        for n2 in range(n1, len(irreps)):
            if PRINT:
                print(f"G[{irreps[n1]-1}/2,{irreps[n2]-1}/2] = {G[n1,n2]}")
            assert abs(G[n1,n2] - G[n2,n1]) < 1e-10


def print_data(data, symbol, label="", precision=3):
    print(label, end='')
    for n in range(_N_):
        if _type_ == complex and np.iscomplexobj(data[0]):
            print(f"{symbol}[{_irreps_[n]-1}/2] = {data[n].real:.{precision}f}+{data[n].imag:.{precision}f}i ", end="\t")
        else:
            print(f"{symbol}[{_irreps_[n]-1}/2] = {data[n]:.{precision}f} ", end="\t")
    print("")


def print_everything(label=""):
    print("\n\n")
    if label != "":
        print(label)
    print(f"Number of irreps: _N_ = {_N_}")
    print("Irreps:")
    for n in range(len(_irreps_)):
        print(f"j[{n}] = {_irreps_[n]-1}/2, dim={_irreps_[n]}", end=" \t")
    print("\nOrbital energies:")
    for n in range(len(_epsilon_)):
            print(f"ε[{n}] = {_epsilon_[n]:.3f}", end=" \t")
    print("\nInteraction matrix elements:")
    for n in range(len(_G_)):
        for m in range(len(_G_[n])):
            print(f"G[{n},{m}] = {_G_[n,m]:.3f}", end=" \t")
        print("")
    print(f"Chemical potential: μ = {_mu_}", end=" \t")
    print(f"Number of nucleons: _Np_ = {_Np_}")
    print("Hamilton matrix elements:")
    for n in range(len(_matrices_)):
        print(f"h[{n}]:")
        for i in range(len(_matrices_[n])):
            for j in range(len(_matrices_[n][i])):
                if abs(_matrices_[n][i,j]) > 1e-8:
                    print(f"\t[{i},{j}] = {_matrices_[n][i,j]:.3f}", end=" \t")
            print("")
    print("\n\n")


def get_parameters_for_Pb(irrep_to_decrement=-1):

    irreps = np.loadtxt('deg.txt', dtype=int)
    epsilon = np.loadtxt('epsilon.txt')
    G = _V0_*np.loadtxt('G.txt')

    if irrep_to_decrement != -1:
        if irreps[irrep_to_decrement] > 2:
            irreps[irrep_to_decrement] -= 2
        else:
            irreps = np.delete(irreps, irrep_to_decrement)
            epsilon = np.delete(epsilon, irrep_to_decrement)
            G = np.delete(G, irrep_to_decrement, 0)
            G = np.delete(G, irrep_to_decrement, 1)
            
    for j in range(len(epsilon)):
        epsilon[j] = epsilon[j]-0.5*G[j,j]
    
    return irreps, epsilon, G


def get_parameters_for_Richardson(irrep_to_decrement=-1):

    irreps = np.zeros(shape=(_L_), dtype=int)
    epsilon = np.zeros(shape=(_L_))
    G = _V0_*np.ones(shape=(_L_,_L_))
    
    for j in range(_L_):
        irreps[j] = 2
        epsilon[j] = j
    
    if irrep_to_decrement != -1:
        irreps = np.delete(irreps, irrep_to_decrement)
        epsilon = np.delete(epsilon, irrep_to_decrement)
        G = np.delete(G, irrep_to_decrement, 0)
        G = np.delete(G, irrep_to_decrement, 1)
    
    #for j in range(_L_):
        #epsilon[j] -= 0.5*G[j,j]
    
    return irreps, epsilon, G


def get_test_parameters():
    global _Np_
    _Np_ = 3
    irreps = np.array([4])
    epsilon = np.array([-1])
    G = np.array([[0.1]])
    print("Using test parameters!")
    return irreps, epsilon, G


def get_parameters_for_Sn(irrep_to_decrement=-1):
    irreps = np.zeros(shape=(5), dtype=int)
    epsilon = np.zeros(shape=(5))
    V = np.zeros(shape=(5,5))
    G = np.zeros(shape=(5,5))

    irreps[0] = 8
    irreps[1] = 6
    irreps[2] = 2
    irreps[3] = 12
    irreps[4] = 4

    V[0,0] = 0.9850
    V[0,1] = 0.5711
    V[0,2] = 0.2920
    V[0,3] = 1.1454
    V[0,4] = 0.5184

    V[1,0] = 0.5711
    V[1,1] = 0.7063
    V[1,2] = 0.3456
    V[1,3] = 0.9546
    V[1,4] = 0.9056

    V[2,0] = 0.2920
    V[2,1] = 0.3456
    V[2,2] = 0.7244
    V[2,3] = 0.4265
    V[2,4] = 0.3515

    V[3,0] = 1.1454
    V[3,1] = 0.9546
    V[3,2] = 0.4265
    V[3,3] = 1.0599
    V[3,4] = 0.6102

    V[4,0] = 0.5184
    V[4,1] = 0.9056
    V[4,2] = 0.3515
    V[4,3] = 0.6102
    V[4,4] = 0.4063
    
    epsilon[0] = -6.121 #g
    epsilon[1] = -5.508 #D
    epsilon[2] = -3.891 #s
    epsilon[3] = -3.778 #h
    epsilon[4] = -3.749 #d

    if irrep_to_decrement != -1:
        if irreps[irrep_to_decrement] > 2:
            irreps[irrep_to_decrement] -= 2
        else:
            irreps = np.delete(irreps, irrep_to_decrement)
            epsilon = np.delete(epsilon, irrep_to_decrement)
            V = np.delete(V, irrep_to_decrement, 0)
            V = np.delete(V, irrep_to_decrement, 1)
            G = np.zeros(shape=(len(irreps),len(irreps)))
	
    for n in range(len(irreps)):
        for m in range(len(irreps)):
            G[n,m] = V[n,m] * _V0_ * 2. / np.sqrt(irreps[n]*irreps[m])

    for n in range(len(irreps)):
        epsilon[n] -= 0.5 * G[n,n]
        
    return irreps, epsilon, G


def deviation_from_particlenumber(mu):
    update_matrices_diagonal(mu)
    self_consistent_diagonalization(max_iterations=200, PRINT=False)
    return _Np_ - sum(calculate_occupation())


def odd_nucleon_number_fixed_shell_and_mu(mu, shell, PRINT=False):
    def incl_sign(x, precision=3):
        if x < 0:
            return f"- {abs(x):.{precision}f}"
        else:
            return f"+ {abs(x):.{precision}f}"
    assert _Np_ % 2 == 1
    if PRINT:
        print("Ground state calculation for shell %i and odd particle number %i" % (shell, _Np_))
    Delta = np.zeros(shape=(_N_))
    occupation = np.zeros(shape=(_N_))
    if _irreps_[shell] == 2:
        initialize(mu, irrep_to_decrement=shell)
        Delta_temp = self_consistent_diagonalization(max_iterations=1000, PRINT=PRINT)
        occupation_temp = calculate_occupation()
        Delta[:shell-1] = Delta_temp[:shell-1]
        Delta[shell] = 0
        Delta[shell+1:] = Delta_temp[shell:]
        occupation[:shell-1] = occupation_temp[:shell-1]
        occupation[shell] = 1
        occupation[shell+1:] = occupation_temp[shell:]
    else:
        initialize(mu, irrep_to_decrement=shell)
        Delta = self_consistent_diagonalization(max_iterations=1000, PRINT=PRINT)
        occupation = calculate_occupation()
        occupation[shell] += 1
    energy = calculate_energy() + _epsilon_[shell]
    initialize(mu)
    return Delta, energy, occupation


def find_lowest_shell_odd_nucleon_number_fixed_mu(mu, PRINT=False):
    def incl_sign(x, precision=3):
        if x < 0:
            return f"- {abs(x):.{precision}f}"
        else:
            return f"+ {abs(x):.{precision}f}"
    global _Np_
    assert _Np_ % 2 == 1
    if PRINT:
        print("Finding energetically favourable shell for odd particle number %i" % _Np_)
    update_matrices_diagonal(mu)
    energies = np.zeros(shape=(_N_))
    Deltas = np.zeros(shape=(_N_,_N_))
    occupations = np.zeros(shape=(_N_,_N_))
    dim_2_irreps = [n for n in range(_N_) if _irreps_[n] == 2]
    for n in range(_N_):
        if PRINT:
            print(f"\nConsider shell {n} with j={_irreps_[n]-1}/2:")
        initialize(mu=mu, irrep_to_decrement=n)
        Delta = self_consistent_diagonalization(max_iterations=1000, PRINT=PRINT)
        energies[n] = calculate_energy() + _epsilon_[n]
        occupation = calculate_occupation()
        if n in dim_2_irreps:
            Deltas[n,:n-1] = Delta[:n-1]
            Deltas[n,n] = 0
            Deltas[n,n+1:] = Delta[n:]
            occupations[n,:n-1] = occupation[:n-1]
            occupations[n,n] = 1
            occupations[n,n+1:] = occupation[n:]
        else:
            Deltas[n] = Delta
            occupations[n] = occupation
            occupations[n,n] += 1
        initialize(mu)
        if PRINT:
            print(f"\tEnergy E = E'[{_irreps_[n]-1}/2] + ε[{_irreps_[n]-1}/2] = {energies[n]-_epsilon_[n]:.3f} {incl_sign(_epsilon_[n])} = {energies[n]:.3f}")
            print_data(Deltas[n], "Δ", "\t")
    n_opt = np.argmin(energies)
    if PRINT:
        print(f"\nLowest: Shell {n_opt} with j={_irreps_[n_opt]-1}/2")
    return Deltas[n_opt], energies[n_opt], occupations[n_opt]


def optimize_for_odd_nucleon_number(Np=-1, PRINT=False):
    def incl_sign(x, precision=3):
        if x < 0:
            return f"- {abs(x):.{precision}f}"
        else:
            return f"+ {abs(x):.{precision}f}"
    global _Np_
    if Np != -1:
        _Np_ = Np
    assert _Np_ % 2 == 1
    if PRINT:
        print("Optimization of chemical potential and shell for odd particle number %i" % _Np_)
    energies = np.zeros(shape=(_N_))
    Deltas = np.zeros(shape=(_N_,_N_))
    mu_opt = np.zeros(shape=(_N_))
    occupations = np.zeros(shape=(_N_,_N_))
    dim_2_irreps = [n for n in range(_N_) if _irreps_[n] == 2]
    _Np_ -= 1
    for n in range(_N_):
        if PRINT:
            print(f"\nConsider shell {n} with j={_irreps_[n]-1}/2:")
        initialize(irrep_to_decrement=n)
        mu_opt[n] = bisect(deviation_from_particlenumber, -20, 20)
        update_matrices_diagonal(mu_opt[n])
        Delta = self_consistent_diagonalization(max_iterations=1000, PRINT=PRINT)
        occupation = calculate_occupation()
        energies[n] = calculate_energy() + _epsilon_[n]
        if n in dim_2_irreps:
            Deltas[n,:n-1] = Delta[:n-1]
            Deltas[n,n] = 0
            Deltas[n,n+1:] = Delta[n:]
            occupations[n,:n-1] = occupation[:n-1]
            occupations[n,n] = 1
            occupations[n,n+1:] = occupation[n:]
        else:
            Deltas[n] = Delta
            occupations[n] = occupation
            occupations[n][n] += 1
        initialize()
        if PRINT:
            print(f"\tEnergy E = E'[{_irreps_[n]-1}/2] + ε[{_irreps_[n]-1}/2] = {energies[n]-_epsilon_[n]:.3f} {incl_sign(_epsilon_[n])} = {energies[n]:.3f}")
            print_data(Deltas[n], "Δ", "\t")
            print_data(occupations[n], "N", "\t")
    _Np_ += 1
    n_opt = np.nanargmin(energies)
    if PRINT:
        print(f"\nLowest: Shell {n_opt} with j={_irreps_[n_opt]-1}/2")
    return Deltas[n_opt], energies[n_opt], occupations[n_opt], mu_opt[n_opt]


def optimize_for_even_nucleon_number(Np=-1, PRINT=False):
    global _Np_
    if Np != -1:
        _Np_ = Np
    if PRINT:
        print("Optimization of chemical potential for even particle number %i" % _Np_)
    assert _Np_ % 2 == 0
    mu_opt = bisect(deviation_from_particlenumber, -20, 20)
    update_matrices_diagonal(mu_opt)
    Delta = self_consistent_diagonalization(max_iterations=1000, PRINT=PRINT)
    occupation = calculate_occupation()
    energy = calculate_energy()
    return Delta, energy, occupation, mu_opt
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-save_Nmu', action='store_true', default=False)
    parser.add_argument('-run_V0', action='store_true', default=False)
    parser.add_argument('-run_Delta3', action='store_true', default=False)
    parser.add_argument('-run_N', action='store_true', default=False)
    parser.add_argument('-V0', type=float, default=1)
    parser.add_argument('-Np', type=int, default=14)
    parser.add_argument('-EL', default='Sn')
    parser.add_argument('-L', default=8, type=int)
    args = parser.parse_args()
    _V0_ = args.V0
    _Np_ = args.Np
    _EL_ = args.EL
    _L_ = args.L
    
    initialize()

    if args.save_Nmu:
        mu_vals = np.linspace(-15.,0.,501,endpoint=True)
        N_vals = []
        for mu_val in mu_vals:
            update_matrices_diagonal(mu_val)
            Delta = self_consistent_diagonalization(max_iterations=1000, PRINT=False)
            occupation = calculate_occupation()
            N_vals.append(sum(occupation))
        np.savetxt("N_mu_V0="+str(_V0_)+"_EL="+_EL_+".dat", np.array([mu_vals, N_vals]).T)

    elif args.run_V0:
        print("Was soll diese Methode können? Der Befehl nj_vals.append(density[_Np_//2]) scheitert.")
        assert False
        V0_vals = np.linspace(0.,1.5,101,endpoint=True)
        nj_vals = []
        for V0_val in V0_vals:
            _V0_ = V0_val
            initialize()
            mu_opt = bisect(deviation_from_particlenumber, -20., 20.)
            update_matrices_diagonal(mu_opt)
            self_consistent_diagonalization(max_iterations=30, PRINT=False)
            occupation = calculate_occupation()
            density = np.array([occupation[n]/_irreps_[n] for n in range(_N_)])
            nj_vals.append(density[_Np_//2])
        np.savetxt("n_G.dat", np.array([V0_vals, nj_vals]).T)

    elif args.run_Delta3:
        if _EL_ == 'Pb': 
            Nmin = 14
            Nmax = 44
        elif _EL_ == 'Sn':
            Nmin = 2
            Nmax = 31
        elif _EL_ == 'Richardson':
            Nmin = 2
            Nmax = 2*_L_-1
        Delta3s = np.zeros(shape=(Nmax+1-Nmin))
        Nshells = np.zeros(shape=(Nmax+1-Nmin))
        for Nshell in range(Nmin,Nmax+1):
            _Np_ = Nshell
            if Nshell % 2 == 1:
                Delta, energy, occupation, mu_opt = optimize_for_odd_nucleon_number(Nshell, False)
                update_matrices_diagonal(mu_opt)
                _Np_ = Nshell - 1
                Delta_m1 = self_consistent_diagonalization(max_iterations=1000, PRINT=False)
                energy_m1 = calculate_energy()
                _Np_ = Nshell + 1
                Delta_p1 = self_consistent_diagonalization(max_iterations=1000, PRINT=False)
                energy_p1 = calculate_energy()
            else:
                Delta, energy, occupation, mu_opt = optimize_for_even_nucleon_number(Nshell, False)
                update_matrices_diagonal(mu_opt)
                _Np_ = Nshell - 1
                Delta_m1 = find_lowest_shell_odd_nucleon_number_fixed_mu(mu_opt, False)
                energy_m1 = calculate_energy()
                _Np_ = Nshell + 1
                Delta_p1 = find_lowest_shell_odd_nucleon_number_fixed_mu(mu_opt, False)
                energy_p1 = calculate_energy()
            Nshells[Nshell-Nmin] = Nshell
            Delta3s[Nshell-Nmin] = energy - 0.5*energy_m1 - 0.5*energy_p1
        np.savetxt("Delta3"+_EL_+".dat", np.array([Nshells, Delta3s]).T)
    
    elif args.run_N:
        
        plot_N = []
        plot_E0 = []
        
        if _EL_ == 'Sn':
            
            Nmin = 2
            Nmax = 32
            
        elif _EL_ == 'Pb':
            
            Nmin = 2
            Nmax = 102
            
        for N in range(Nmin,Nmax+1):
            _Np_ = N
            if _Np_ % 2 == 0:
                Delta, energy, occupation, mu_opt = optimize_for_even_nucleon_number(PRINT=True)
            else:
                Delta, energy, occupation, mu_opt = optimize_for_odd_nucleon_number(PRINT=True)
            plot_N.append(N)
            plot_E0.append(energy)
        for i in range(len(plot_N)):
            print(plot_N[i],"\t",plot_E0[i])
        
    else:
        #_Np_ = 14
        if _Np_ % 2 == 0:
            Delta, energy, occupation, mu_opt = optimize_for_even_nucleon_number(PRINT=True)
        else:
            Delta, energy, occupation, mu_opt = optimize_for_odd_nucleon_number(PRINT=True)
        print("\nResult of self-consistent groundstate calculation:\n")
        print(f"Ground state energy:  \tE_0 = {energy}")
        print(f"chemical potential:  \t\tμ = {mu_opt}")
        print(f"Nucleon number:   \t\tN = {sum(occupation)}")
        print_data(Delta, 'Δ', "Pairing gaps:\t")
        print_data(occupation, 'N', "Occupation: \t\t")
        print_data(np.array([occupation[n]/_irreps_[n] for n in range(_N_)]), 'n', "Density: \t\t")
    
