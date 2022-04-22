#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Computational physics: project 2
To do the analysis of the simulation to obtain correlation time, the average magnetization per
spin, average energy per spin, magnetic susceptibility and specific heat.
"""

import numpy as np
import matplotlib.pyplot as plt

global N_grid, kB, num_time_steps, eng_path, mag_path


def getchi(mag, tmax):
    """To calculate the correlation function of the magnetization.
    Args:
        mag (array): the magnetization per time step
        tmax (scalar): the total number of time steps of the simulation
    Returns:
        chi (array): the correlation function"""
    chi = np.zeros(tmax)
    for t in range(tmax):
        sum1, sum2, sum3 = 0, 0, 0
        for i in range(tmax - t): #i is t'
        	#print(t,i)
        	sum1 += mag[i]*mag[i+t]
        	sum2 += mag[i]
        	sum3 += mag[i+t]
        chi[t] = 1/(tmax-t)*(sum1 - (sum2*sum3)/(tmax-t))
    return chi

def correlationtime(chi):
    """To calculate the correlation time.
    Args:
        chi (array): the correlation function
    Returns:
        tau (scalar): the correlation time"""
    tau = 0 
    t = 0
    while chi[t] > 0:
        tau += chi[t]/chi[0]
        t += 1
    return tau
    
def stdev(tau, ave2, ave, tmax):
    """To calculate the standard deviation of the average magnetization and energy.
    Args:
        tau (scalar): the correlation time
        ave2 (scalar): the average of the squared
        ave (scalar): the average
        tmax (scalar): the maximum number of timesteps
    Returns:
        the standard deviation (scalar)"""
    return np.sqrt(2*tau/tmax*(ave2 - ave**2))

def get_meanspin(mag, tau, tmax):
    """To calculate the mean magnetization per spin
    Args:
        mag (array); the magnetization of the system
        tau (scalar): the correlation time
        tmax (scalar): the maximum number of timesteps
    Returns:
        mean_spin (scalar): the average magnetization per spin
        std (scalar): the standard deviation of the magnetization"""
    
    mean_spin_signed= np.average(mag[::2*int(np.ceil(tau))])/(N_grid**2)
    mean_spin = np.average(abs(mag[::2*int(np.ceil(tau))])/(N_grid**2))
    mean_spinsqrd = np.average((mag[::2*int(np.ceil(tau))]/(N_grid**2))**2)
    std = stdev(tau, mean_spinsqrd, mean_spin, tmax)
    
    return mean_spin_signed, mean_spin, std

def get_meanenergy(en, tau, tmax):
    """To calculate the mean energy per spin
    Args:
        en (array); the energy of the system
        tau (scalar): the correlation time
        tmax (scalar): the maximum number of timesteps
    Returns:
        mean_en (scalar): the average energy per spin
        std (scalar): the standard deviation of the magnetization"""
    mean_en = np.average(en[::2*int(np.ceil(tau))]/(N_grid**2))
    mean_ensqrd = np.average((en[::2*int(np.ceil(tau))]/(N_grid**2))**2)
    std = stdev(tau, mean_ensqrd, mean_en, tmax)
    
    return mean_en, std


def get_magsus(mag, tau, beta, tmax):
    """To calculate the mean energy per spin
    Args:
        en (array); the energy of the system
        tau (scalar): the correlation time
        tmax (scalar): the maximum number of timesteps
    Returns:
        mean_en (scalar): the average energy per spin
        std (scalar): the standard deviation of the energy"""
    sus = np.zeros(int(tmax/(16*tau)))
    if int(tmax/(16*tau)) > 0:
    	for t in range(int(tmax/(16*tau))):
    		mean_spin = np.average(mag[t*int(16*tau):(t+1)*int(16*tau):2*int(np.ceil(tau))])
    		mean_spinsqrd = np.average((mag[t*int(16*tau):(t+1)*int(16*tau):2*int(np.ceil(tau))])**2)
    		sus[t] = beta/(N_grid**2)*(mean_spinsqrd-mean_spin**2)
    	ave_sus = np.average(sus)
    	std = np.std(sus)
    else:
    	print('Not enough timesteps to calculate magnetic susceptibility')
    	ave_sus = np.nan
    	std = 0
    
    return ave_sus, std
    
def get_specheat(en, tau, beta, temp, tmax):
    """To calculate the average heat capacity
    Args:
        en (array); the energy of the system
        tau (scalar): the correlation time
        beta (scalar): the inverse temperature
        temp (scalar): the temperature
        tmax (scalar): the maximum number of timesteps
    Returns:
        ave_Cv (scalar): the average heat capacity
        std (scalar): the standard deviation of the average"""
    Cv = np.zeros(int(tmax/(16*tau)))
    if int(tmax/(16*tau)) > 0:
    	for t in range(int(tmax/(16*tau))):
    		mean_en= np.average(en[t*int(16*tau):(t+1)*int(16*tau):2*int(np.ceil(tau))])
    		mean_ensqrd = np.average((en[t*int(16*tau):(t+1)*int(16*tau):2*int(np.ceil(tau))])**2)
    		
    		Cv[t] = beta/(temp*N_grid**2)*(mean_ensqrd-mean_en**2)
    	ave_Cv = np.average(Cv)
    	std = np.std(Cv)
    else:
    	print('Not enough timesteps to calculate Specific Heat')
    	ave_Cv = np.nan
    	std = 0
    return ave_Cv, std

def run_analysis(temps):
    """The actual analysis
    Args:
        temps (array): the temperatures of the simulations
        Hfield (array): the magnetic field value of the simulations
    """
    mean_mag, sigma_mag = np.zeros((len(temps), len(Hfield))), np.zeros((len(temps), len(Hfield)))
    mean_en, sigma_en = np.zeros((len(temps), len(Hfield))), np.zeros((len(temps), len(Hfield)))
    mean_sus, sigma_sus = np.zeros((len(temps), len(Hfield))), np.zeros((len(temps), len(Hfield)))
    mean_Cv, sigma_Cv = np.zeros((len(temps), len(Hfield))), np.zeros((len(temps), len(Hfield)))
    tau = np.zeros((len(temps), len(Hfield)))
    
    mean_mag_signed = np.zeros((len(temps), len(Hfield)))
    
    for i, H in enumerate(Hfield):
        for idx, temp in enumerate(temps):
            beta = 1/(temp*kB)
            print('H:{:.2f} T:{:.2f}'.format(H, temp))
            tmax = num_time_steps
  		
            filename = '{}/T{:.2f}_N{}_H{}_mag.csv'.format(mag_path,temp,N_grid,H)
            mag = np.loadtxt(filename, delimiter =',')
            filename = '{}/T{:.2f}_N{}_H{}_eng.csv'.format(eng_path,temp,N_grid,H)
            en = np.loadtxt(filename, delimiter =',')
  	
            chi = getchi(mag, tmax) 
            tau[idx][i] = correlationtime(chi)
            #print(tau[idx])
  	
            mean_mag_signed[idx][i], mean_mag[idx][i], sigma_mag[idx][i] = get_meanspin(mag, tau[idx][i], tmax)
            mean_en[idx][i], sigma_en[idx][i] = get_meanenergy(en, tau[idx][i], tmax)
            mean_sus[idx][i], sigma_sus[idx][i] = get_magsus(mag, tau[idx][i], beta, tmax)
            mean_Cv[idx][i], sigma_Cv[idx][i] = get_specheat(en, tau[idx][i], beta, temp, tmax)
            
            
        if H == 0: # to avoid too many figures
        	sigma_mag_plt = sigma_mag[:, i]#np.reshape(sigma_mag[i], np.shape(sigma_mag[i])[0])
        	sigma_en_plt = sigma_en[:, i]#np.reshape(sigma_en[i], np.shape(sigma_en[i])[0])
        	sigma_sus_plt = sigma_sus[:, i]#np.reshape(sigma_sus[i], np.shape(sigma_sus[i])[0])
        	sigma_Cv_plt = sigma_Cv[:, i]#np.reshape(sigma_Cv[i], np.shape(sigma_Cv[i])[0])
        	
        	plt.figure()
        	plt.scatter(temps, tau[:,i])
        	plt.xlabel("Temperature (J/$k_B$)") 
        	plt.ylabel(r'$\tau$ (a.u.)')
        	plt.title("Correlation time")
        	plt.show()
        	
        	plt.figure()
        	plt.errorbar(temps, mean_mag[:,i], linestyle ='', marker ='o', yerr= sigma_mag_plt)
        	plt.xlabel("Temperature (J/$k_B$)")
        	plt.ylabel("|m| (a.u.)")
        	plt.title("Average magnetization")
        	plt.show()
        	
        	plt.figure()
        	plt.errorbar(temps, mean_en[:,i], linestyle ='', marker ='o',yerr= sigma_en_plt)
        	plt.xlabel("Temperature (J/$k_B$)")
        	plt.ylabel("e (J)")
        	plt.title("Average energy")
        	plt.show()
        	
        	plt.figure()
        	plt.errorbar(temps, mean_sus[:,i], linestyle ='', marker ='o',yerr= sigma_sus_plt)
        	plt.xlabel("Temperature (J/$k_B$)")
        	plt.ylabel("$\chi_M$ (a.u.)")
        	plt.title("Magnetic susceptibility")
        	plt.show()
        	
        	plt.figure()
        	plt.errorbar(temps, mean_Cv[:,i], linestyle ='', marker ='o', yerr= sigma_Cv_plt)
        	plt.xlabel("Temperature (J/$k_B$)")
        	plt.ylabel("$C_v$ (a.u)")
        	plt.title("Specific heat")
        	plt.show()
    
    plt.figure() 
    for fig in range(len(temps)):
        plt.errorbar(Hfield, mean_mag_signed[fig], yerr = sigma_mag[fig], linestyle = '', marker = 'o', label = ' T = {:.2f}'.format(temps[fig]))#yerr = sigma_mag[fig],, color = 'k'
   
    plt.legend()
    plt.xlabel("H (a.u.)")
    plt.ylabel("m (a.u.)")
    plt.title("Magnetization under influence of a magnetic field")
    plt.show()
        

def an_set_globals(N_in = 50, kB_in = 1, Hfield_in = [0], num_time_steps_in = 4000, eng_path_in='./eng_output', mag_path_in='./mag_output'):
    """To initialize the variables of the analysis
        Args:
            N_in (scalar): the number of spins
            kB_in (scalar): the Boltzmann constant
            Hfield_in (scalar): the magnetic field strength
            num_time_steps_in (scalar): the number of time steps
            eng_path_in (string): the directory to save the energy data
            mag_path_in (string): the directory to save the magnetization data
    """
    global N_grid, kB, num_time_steps, eng_path, mag_path, Hfield
	
    N_grid = N_in
    num_time_steps = num_time_steps_in
    kB = kB_in
    eng_path = eng_path_in
    mag_path = mag_path_in
    Hfield = Hfield_in
	
    return
