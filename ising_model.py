"""Computational physics: project 2
The code to simulate the Ising Model."""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, jit
from tqdm import tqdm

global N_grid, kB, J_int, Hfield, T_crit, num_time_steps, verbose, eng_path, mag_path, plot_grid

plot_grid = False #For debugging purposes. If you want to plot (with imshow) the system grid after the simulation has run

def sim_set_globals(N_in=50, kB_in=1, J_in=1, H_in=[0], T_crit_in=2.269, num_time_steps_in = 4000, verbose_in=False, eng_path_in='./eng_output', mag_path_in='./mag_output'):
	'''to be called by the wrapper, sets the values of the global variables for the simulation
	Args:
        N_grid (scalar) = array dimension
    	J_int (scalar) = interaction energy in dimensionless units
    	Hfield (scalar) = magnetic field in dimensionless units
    	T_crit (scalar) = critical temperature (set to 0 if you want all arrays initialized randomly)
    	num_time_steps (scalar) = number of time steps to run for
    	verbose (true/false) = print output for each possible flip (not recommended for N>10)
    	eng_path (string) = path to output directory to store energy data
    	mag_path (string) = path to output directory to store magnetic data
    	
	(the contents of eng_path and mag_path are overwritten each time the code is run, they can 
	be the same directory)
	'''
	global N_grid, kB, J_int, Hfield, T_crit, num_time_steps, verbose, eng_path, mag_path
	
	N_grid = N_in
	J_int = J_in
	Hfield = H_in
	T_crit = T_crit_in
	verbose = verbose_in
	num_time_steps = num_time_steps_in
	kB = kB_in
	eng_path = eng_path_in
	mag_path = mag_path_in
	
	return


@jit(forceobj=True)
def create_system(N_grid, temp, H_curr):
	'''initializes an N by N system array. If the system temperature (T), is less than the 
	expected critical temperature, initializes to all -1 or all 1. Otherwise, initializes 
	randomly
    Args:
        N_grid (scalar): array dimension
        temp (scalar): the temperature of the system
    Returns:
        system_arr (array): array of spins
        in_en (scalar): the system energy'''
	
	if temp < T_crit:
		system_arr = np.ones((N_grid,N_grid)) * np.random.choice([-1,1])
	else :
		system_arr = np.random.choice([-1,1], size=(N_grid, N_grid))
	#plt_system(system_arr)
	if verbose:
		print('Created System') 
	in_en = calc_tot_energy(system_arr, H_curr)
	
	return system_arr, in_en

@jit(forceobj=True)
def calc_tot_energy(system_arr, H_curr):
	'''Calculates the hamiltonian of the entire input system array. (Slow, should only be
	used on setup
    Args:
        system_arr (array): array of spins
    Returns:
        ham (scalar): the Hamiltonian'''
	int_energy = get_neighbor_sum(system_arr)
	mag_energy = get_mag(system_arr)
	
	ham = -J_int * int_energy - H_curr * mag_energy
	
	return ham
	
@njit
def get_neighbor_sum(system_arr):
	'''Loops over whole system, calculating neighbor sums for energy calculation. (Slow, 
	should only be used on setup
    Args:
        system_arr (array): array of spins
    Returns:
        e/2 (scalar): the change in energy'''
	e = 0
	for i in range(N_grid):
		for j in range(N_grid):
			e += (system_arr[i,j]*system_arr[(i-1)%N_grid,j] + system_arr[i,j]*system_arr[(i+1)%N_grid,j] + system_arr[i,j]*system_arr[i,(j+1)%N_grid] + system_arr[i,j]*system_arr[i,(j-1)%N_grid])
	return(e/2) #divide by two so we don't double-count pairs

@njit
def get_mag(system_arr):
	'''Returns the total magnetization of the system
    Args:
        system_arr (array): array of spins
    Returns:
        the total magnetization'''
	return(np.sum(system_arr))

@njit
def get_next_state(system_arr):
	'''Flips a random spin in system_arr and returns the new system and position (i,j) of
	the changed spin. In doing so, w is implicitly enforced
    Args:
        system_arr (array): array of spins
    Returns:
        next_state (array): array with one spin flipped compared to the previous array
        i, j (scalar): the position of the flipped spin'''
	next_state = np.copy(system_arr)
	
	i = np.random.randint(0, N_grid) #implicitly enforcing w
	j = np.random.randint(0, N_grid) #only one spin is flipped w/ uniform probability
	next_state[i,j] *= -1
	
	return next_state, i, j

@jit(forceobj=True)
def calc_A(system_arr, next_state, i, j, beta, curr_e, H_curr):
	'''Calculates and returns A (the probability of transition) between the current state 
	and the next possible state. Takes in the current array, the next state array, the 
	location (i,j) of the flipped spin, beta, and the current state's energy
    Args:
        system_arr (array): the old state
        next_state (array): the new state
        i, j (scalars): the position of the flipped spin
        beta (scalar): inverse temperature
        curr_e (scalar): the energy of the configuration in system_arr
    Returns:
        Aprob (scalar): the probability for the system to go to the new configuration
        new_e (scalar): the energy of the new state'''
	
	new_e = update_energy(system_arr, i, j, curr_e, H_curr) #calculates new energy
	
	if new_e < curr_e: #if the new energy is less than the current, we always transition
		Aprob = 1.
		
	else: #otherwise, calculate the probability
		Aprob = get_prob(new_e, curr_e, beta)
		
	if verbose:
		print('Current System Energy: {}'.format(curr_e))
		print('New State Energy: {}'.format(new_e))
		#print('A = {}'.format(A))
	
	return Aprob, new_e
	
@njit
def get_prob(new_e, curr_e, beta):
    '''Calculates and returns the probability of transition from one state to another based 
    on their energies and the temperature of the system (implicitly stored in beta) 
    Args:
        new_e (scalar): the energy of the new state
        curr_e (scalar): the energy of the old state
        beta (scalar): inverse temperature
    Returns:
        prob (scalar): the Boltzmann factor for the new state'''
    prob = np.exp(beta*(curr_e - new_e)) #using exponent rules, no need to use a 128-bit float
    
    return prob

@jit(forceobj=True)
def change_state(Aprob):
	'''Based on the probability of transition (A), randomly chooses if the system changes
	state. Returns a boolean.
    Args: 
        Aprob (scalar): the probability of the transition'''
	return(np.random.choice([True,False], p=[Aprob, 1-Aprob]))
	
@njit
def update_energy(system_arr, i, j, curr_en, H_curr):
	'''Calculates the change in energy of a system if it were to experience a spin-flip at 
	(i,j), and returns the new system energy.
	Updates from the current energy instead of recalculating over the whole array for speed.
    Args:
        system_arr (array): array of spins
        i, j (integer): the position of the flipped spin
        curr_en (scalar): the energy of the configuration in system_arr
    Returns: the updated energy (scalar)'''
	change_en = (system_arr[i,j]*system_arr[(i-1)%N_grid,j] + system_arr[i,j]*system_arr[(i+1)%N_grid,j] + system_arr[i,j]*system_arr[i,(j+1)%N_grid] + system_arr[i,j]*system_arr[i,(j-1)%N_grid])
	
	return curr_en + 2*change_en + 2*H_curr*system_arr[i,j]#2x -> one to get to a "neutral" state, one to actually flip

@jit(forceobj=True)
def update_system(system_arr, beta, curr_e, H_curr):
	'''Finds a new state for the system, the probability of switching into that state, and
	switches as necessary
    Args:
        system_arr (array): array of spins
        beta (scalar): the inverse temperature)
        curr_e (scalar): the energy of system_arr
    Returns:
        system_arr: the new state. (The previous variable is overwritten if the state is changed)
        curr_e (scalar): the new energy. (The previous variable is overwritten if the state is changed)'''
	next_state, i, j = get_next_state(system_arr)
	Aprob, new_e = calc_A(system_arr, next_state, i,j, beta, curr_e, H_curr)
	
	if change_state(Aprob): #if moving to new state
		if verbose:
			print('Changing state...\n')
		curr_e = new_e
		system_arr = np.copy(next_state)
	else:
		if verbose:
			print('Retaining state...\n')
	
	return system_arr, curr_e

def plt_system(system_arr):
	'''plots the system using imshow
    Args:
        system_arr (array): array of spins'''
	plt.imshow(system_arr)
	plt.show()
	return

@jit(forceobj=True)
def run(temps):
    '''run the simulation at the given temperature(s)
    Args:
        temps (list): list of simulation temperatures'''
    for H_curr in Hfield:
    	print('H: {:.2f}'.format(H_curr))
    	for T_idx, temp in enumerate(temps):
    		print('Temp: {:.2f}'.format(temp))
    		beta = 1/(temp*kB)
    		Mag = np.zeros(num_time_steps) #magnetization at this temp over time
    		Eng = np.zeros(num_time_steps) #energy at this temp over time
    		system_arr, curr_e = create_system(N_grid, temp, H_curr)  #initialize
    		#print(curr_e)
    		
    		for time_step in tqdm(range(num_time_steps)):
    			for i in range(N_grid**2): #one "sweep" of the array per time step
    				system_arr, curr_e = update_system(system_arr, beta, curr_e, H_curr) #take a step
    				Mag[time_step] = get_mag(system_arr) #save magnetization
    				Eng[time_step] = curr_e #save energy
    		if plot_grid:
    			plt_system(system_arr)
    		np.savetxt('{}/T{:.2f}_N{}_H{}_mag.csv'.format(mag_path,temp,N_grid,H_curr), Mag, delimiter=",") #output to files for analysis
    		np.savetxt('{}/T{:.2f}_N{}_H{}_eng.csv'.format(eng_path,temp,N_grid,H_curr), Eng, delimiter=",")
    return

