# COP Assignment 2
### Fall 2022
### Marieke Visscher and Catherine Slaughter

## Description

An easy-to-use MCMC simulation of the 2D Ising model.

## Dependencies

This code uses a few relatively basic packages that should be easily installable via your preferred python package management method (conda, pip, etc.)
- numpy
- matplotlib
- numba 
- tqdm
- os

## Usage

The easiest way to run this simulation code is through the provided wrapper (run_simulation.py). This piece of code allows the user to change parameter values, input a list of model temperatures, and to pick which pieces of code to be run. The available input parameters are:
- N_grid: the dimensions of the simulated array. Integer
- J_int: the strength of spin-spin interactions. Integer
- Hfield: the magnetic field strength(s) to test. List of floats
- T_crit: the expected critical temperature. Models with temperatures below this value are entirely initialized to +/-1, others are initialized with random spin directions. Set T_crit = 0 to initialize all models randomly. Float
- verbose: tells simulation code whether to print out the calculated energies and state transitions. **For debugging purposes. Recommended to remain "False".** Boolean
- num_time_steps: the number of time steps to simulate. Integer
- kB: Boltzmann constant. Equal to 1 in dimensionless units. Float
- eng_path: Path to directory where energy data is saved. Files inside this directory should be expected to be overwritten. String
- mag_path: Path to directory where magnetization data is saved. Files inside this directory should be expected to be overwritten. String
	- note that eng_path and mag_path can be the same, if the user would like.
- temps: the model temperature(s), must be an iterable form (list, np array, etc.). Floats

Telling the wrapper which code(s) to run is done via three Boolean variables. For each, a value of True means the corresponding code will be run, and False means it will not. These control variables are:

- run_new: Runs ising_model.py. Creates and runs a new simulation with the given input parameters, save data at eng_path and mag_path
- analyse: Runs analysis.py. Run the analysis for a simulation with given input parameters and output data stored at eng_path and mag_path
- plot_mag: Runs plot_mag.py. Plots the average magnetization per spin over time for the first 16 temperatures given.
