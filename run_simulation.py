from ising_model import *
from analysis import *
from plot_mag import *
import os

global N_grid, kB, J_int, H, T_crit, num_time_steps, verbose, eng_path, mag_path

############# USER INPUT ####################
N_grid = 50 #the dimensions of the grid
J_int = 1 #the strength of the spin-spin interactions
Hfield = [0] #the magnetic field. Input as a list of floats
T_crit = 0 #expected critical temperature to initialize spin arrays
verbose = False
num_time_steps = 5000 #total time steps of the simulation
kB = 1 #Boltzmann constant
eng_path = './eng_output' #paths to the directories to save the data
mag_path = './mag_output'

#temperatures
temps = np.arange(1.,4.2,.2)

#codes
run_new = True #run new simulation with given parameters
analyse = True #analyze data w/ outputs in eng_path and mag_path
plot_mag = False #plot avg magnetic vs time for up to 16 models
#############################################

if not os.path.exists(eng_path):
	print('Making Directory {}'.format(eng_path))
	os.mkdir(eng_path)
	
if not os.path.exists(mag_path):
	print('Making Directory {}'.format(mag_path))
	os.mkdir(mag_path)

if run_new:
	sim_set_globals(N_grid, kB, J_int, Hfield, T_crit, num_time_steps, verbose, eng_path, mag_path)
	run(temps)

if analyse:
	if len(os.listdir(mag_path)) == 0:
		print('No output files to analyse in {}'.format(mag_path))
	else:
		an_set_globals(N_grid, kB, Hfield, num_time_steps, eng_path, mag_path) 
		run_analysis(temps)

if plot_mag:
	if len(os.listdir(mag_path)) == 0:
		print('No output files to analyse in {}'.format(mag_path))
	else:
		plt_set_globals(N_grid, mag_path)
		plt_mags(temps)
		

