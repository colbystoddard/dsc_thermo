#!/usr/bin/env python
# coding: utf-8


import numpy as np
from numpy import sin, cos, pi, exp
from numpy import log as ln
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy import integrate
import warnings
import re

import molar_mass
################################################################################

mu_sapphire = molar_mass.molar_mass("Al2O3")
#sapphire_Cp_data = np.genfromtxt("Cp_sapphire.txt", skip_header=3, unpack=True) #from TA instruments
sapphire_Cp_data = np.genfromtxt("Sapphire_Cp_ASTM.txt", skip_header=1, unpack=True) #from ASTM

cp_sapphire = interpolate.interp1d(sapphire_Cp_data[0], mu_sapphire*sapphire_Cp_data[2]) #linear interpolation
#TODO check if this is okay


#find indices of ends of temperature plateaus
#Note: this function assumes the program ends with an isothermal that is not part of the data collection
def get_endpoints(program_temp):
    plateaus = np.where(np.diff(program_temp) == 0)[0] + 1 
    #^ add 1 b/c diff gives arr[i+1]-arr[i], but we need arr[i]-arr[i-1]
    endpoints = plateaus[np.where(np.diff(plateaus) != 1)[0]]
    return endpoints

#determine if equilibrium was reached for each isotherm
def equilibrium_reached(sample, N_points=100, tol=0.01, use_derivative=False):
    endpoints = get_endpoints(sample[3])
    
    in_equilibrium = np.ones(len(endpoints), dtype=bool)
    if use_derivative:
        for isotherm, endpoint in enumerate(endpoints):
            #using unsubtracted heat flow
            in_equilibrium[isotherm] = np.all(abs((sample[1][endpoint]-sample[1][endpoint-N_points])/(sample[0][endpoint]-sample[0][endpoint-N_points])/60) < tol)
    else:
        for isotherm, endpoint in enumerate(endpoints):
            #using unsubtracted heat flow
            in_equilibrium[isotherm] = np.all(abs(sample[1][endpoint-N_points:endpoint] - sample[1][endpoint]) < tol)        
        
    return in_equilibrium

#gets sample weight and measurement data from dsc output file  
def read_dsc_output(filename):
    if ".txt" in filename:
        with open(filename, "rb") as file:
            line = file.readline()
            while line != b'':
                line = file.readline()
                if b"Sample Weight" in line:
                    *_, weight, units = line.split()
                    if units == b"mg":
                        weight = 1e-3*float(weight)
                    else:
                        raise RuntimeError("Unknown mass units")

                if ("\t" +" "*10 + "\t").encode() in line:
                    break

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")        
                data = np.genfromtxt(file, unpack=True, invalid_raise=False)
                
    elif ".csv" in filename:
        data = np.genfromtxt(filename, delimiter=",", skip_header=1, unpack=True)
        weight = None
    
    return data, weight

'''
plots heatflow vs. time.
    sample: array of sample measurement data (in order of exported ds8d data)
    sample_filters: list of arrays of indices. Each array corresponds to points you wish to be
        highlight
    filter_labels: list of strings that will identify the highlighted points in the legend
        (must be specified to plot highlighted points)
    filter_colors: list of colors of the highlighted points. See matplotlib.pyplot.plt
    axes: axes to plot data on (if None, new axes will be created)

    returns: axes used to plot data
'''
def plot_heatflow(sample, sample_filters=[], filter_labels=None, filter_colors=None, axes=None):
    if filter_colors is None:
        filter_colors = ["C{}".format(i+1) for i in range(len(sample_filters))]
    if axes is None:
        fig, ax = plt.subplots(figsize=(8,5))
        ax2 = ax.twinx()
        axes = (ax, ax2)
    else:
        ax, ax2 = axes

    y1 = min(sample[1])*np.ones(len(sample[1])-1)
    y2 = max(sample[1])*np.ones(len(sample[1])-1)
    
    ax.fill_between(sample[0,:-1], y1, y2, where=[np.diff(sample[3])>0][0], color="gray", alpha=0.3)
    ax.plot(sample[0], sample[1], "C0")

    ax2.plot(sample[0], sample[4], "k")
    
    if filter_labels is not None:
        for sample_filter, filter_label, filter_color in zip(sample_filters, filter_labels, filter_colors):
            ax.plot(sample[0][sample_filter], sample[1][sample_filter], 
                    ".", ms=10, color=filter_color, label=filter_label)
        ax.legend(loc="lower right")

    else:
        for sample_filter, filter_color in zip(sample_filters, filter_colors):
            ax.plot(sample[0][sample_filter], sample[1][sample_filter], ".", color=filter_color, ms=10)

    ax.set(xlabel="Time (s)", ylabel="Heat Flow (mW)")
    ax2.set(ylabel="Temperature ($^\circ C$)")
    return axes

'''
Finds the data points to use for dQ/dT|_{dT/dt \neq 0} in the cp calculation. Currently uses the highest 
valid temperature below each isotherm, where valid means the points are in the temperature ranges that
dQ/dT|_{dT/dt \neq 0} for the sapphire/empty measuements can be interpolated. 
    sample/sapphire/empty: array of sample/sapphire/empty measurement data from dsc output
    
    returns:
        array of indices corresponding to points to use in the cp calculation
'''
def get_filter(sample, sapphire, empty):
    #return np.where(np.diff(sample[3]) > 0)[0] + 1
    sample_endpoints = get_endpoints(sample[3])
    sample_incr = np.where(np.diff(sample[3]) > 0)[0] + 1
    sapphire_incr = np.where(np.diff(sapphire[3]) > 0)[0] + 1
    empty_incr = np.where(np.diff(empty[3]) > 0)[0] + 1

    sample_filter = np.zeros(len(sample_endpoints[1:]), dtype=int) 
    for i, endpoint in enumerate(sample_endpoints[1:]):
        max_sapphire = max(sapphire[4][sapphire_incr][sapphire[3][sapphire_incr]<=sample[3][endpoint]])
        max_empty = max(empty[4][empty_incr][empty[3][empty_incr]<=sample[3][endpoint]])
        max_sample = min([max_sapphire, max_empty])

        in_range = np.where(sample[4]<=max_sample)[0]
        
        sample_filter[i] = max(np.intersect1d(sample_incr, in_range))
        
    return sample_filter

'''
Finds heat flow needed to keep T constant for the isotherm just above a given point (finds dQ/dT|_{dT/dt=0}).
The last data point in the isotherm is used to ensure the system has come to equilibrium (assuming the isotherm
was long enough). 
    heat_flow: array of heat flow measurements for the sample
    sample_filter: array of indices corresponding to data points for which this calculation should be done
    endpoints: array of indices of the last data point in each isotherm
    
    returns:
        array of dQ/dT|_{dT/dt=0} corresponding to the necessary data points
'''
def dQdt_flat(heat_flow, sample_filter, endpoints):
    heat_flow_flat = np.zeros(len(heat_flow[sample_filter]))
    for flat_ind, filter_ind in enumerate(sample_filter):
        heat_flow_flat[flat_ind] =  heat_flow[min(endpoints[endpoints>filter_ind])]
    return heat_flow_flat

'''
Interpolate values of Q_dot for a reference material at the temperatures the sample was measured at.
Q_dot is defined in Glade et al. 2000
    sample: array of sample measurement data (in order of exported ds8d data)
    reference: array of reference measurement data " "
    ref_endpoints: indices of ends of temperature plateaus of reference data
    samp_filter: array of indices corresponding to the sample measurements you want to include. If None,
        all sample measurements where the program temperature is increasing will be used.
    
    returns: 
        array of Q_dot values for the reference evaluated at the sample temperatures for 
        points where the sample program temperature is increasing.
'''
def get_Q_dot(sample, reference, ref_endpoints, sample_filter=None):
    if sample_filter is None:
        sample_filter = np.where(np.diff(sample[3]) > 0)[0] + 1
    ref_incr = np.where(np.diff(reference[3]) > 0)[0] + 1
    
    Q_dot = np.zeros(len(sample_filter))
    #using unsubtracted heat flow
    heat_flow_flat = dQdt_flat(reference[1], ref_incr, ref_endpoints)
    
    endpoint_program_temps = reference[3][ref_endpoints]
    
    min_program_temp, max_program_temp = -np.inf, -np.inf
    for Q_dot_ind, samp_ind in enumerate(sample_filter): #only where sample program temp is increasing
        sample_program_temp =  sample[3][samp_ind]
        
        #interpolate using only data between the nearest isotherms
        if sample_program_temp > max_program_temp:
            min_program_temp = max(endpoint_program_temps[endpoint_program_temps<sample_program_temp])
            max_program_temp = min(endpoint_program_temps[endpoint_program_temps>=sample_program_temp])
            
            temp_range = np.where((reference[3][ref_incr] <= max_program_temp) & (reference[3][ref_incr] > min_program_temp))
            
            #using unsubtracted heat flow
            Q_dot_func = interpolate.interp1d(reference[4][ref_incr][temp_range], 
                                      reference[1][ref_incr][temp_range] - heat_flow_flat[temp_range],
                                     fill_value=np.nan, bounds_error=False)
        Q_dot[Q_dot_ind] = Q_dot_func(sample[4][samp_ind])
    return Q_dot



#stores data from dsc measurements and provides methods for calculating physical quantities from this data 
class Heat_Flow_Data:
    def __init__(self, sample_file, sapphire_file, empty_file):
        self.sample, self.sample_mass = read_dsc_output(sample_file)
        self.sapphire, self.sapphire_mass = read_dsc_output(sapphire_file)
        self.empty, _ = read_dsc_output(empty_file)

    #check that the isotherms of the 3 runs are at the same temperature
    def check_isotherms(self, tol=0.2):
        sample_endpoints = get_endpoints(self.sample[3])
        sapphire_endpoints = get_endpoints(self.sapphire[3])
        empty_endpoints = get_endpoints(self.empty[3])

        #only check isotherms where sapphire and empty are supposed to match sample
        sapphire_trim = np.in1d(self.sapphire[3][sapphire_endpoints], self.sample[3][sample_endpoints])
        empty_trim = np.in1d(self.empty[3][empty_endpoints], self.sample[3][sample_endpoints])
        if (abs(self.sample[4][sample_endpoints] - self.sapphire[4][sapphire_endpoints][sapphire_trim]) > tol).any():
            #print(abs(self.sample[4][sample_endpoints]-self.sapphire[4][sapphire_endpoints])>0.1)
            raise RuntimeError("Sample and Sapphire hold at different temperatures")
        elif (abs(self.sample[4][sample_endpoints]-self.empty[4][empty_endpoints][empty_trim]) > tol).any():
            raise RuntimeError("Sample and Empty hold at different temperatures")
    
    #from Glade et al 2000
    def cp(self, mu_samp, plot=False):
        self.check_isotherms()
        sample_endpoints = get_endpoints(self.sample[3])
        sapphire_endpoints = get_endpoints(self.sapphire[3])
        empty_endpoints = get_endpoints(self.empty[3])
        
        sample_filter = get_filter(self.sample, self.sapphire, self.empty)

        T_samp = self.sample[4][sample_filter]
        
        m_samp, m_sapphire = self.sample_mass, self.sapphire_mass
        
        #using unsubtracted heat flow
        Q_dot_samp = self.sample[1][sample_filter] - dQdt_flat(self.sample[1], sample_filter, sample_endpoints)
        Q_dot_sapphire = get_Q_dot(self.sample, self.sapphire, sapphire_endpoints, sample_filter=sample_filter)
        Q_dot_empty = get_Q_dot(self.sample, self.empty, empty_endpoints, sample_filter=sample_filter)
        
        cp_samp = (Q_dot_samp-Q_dot_empty)/(Q_dot_sapphire-Q_dot_empty)*(m_sapphire*mu_samp)/(m_samp*mu_sapphire)*cp_sapphire(T_samp)
        if plot:
            plt.plot(T_samp, (Q_dot_samp-Q_dot_empty)/(Q_dot_sapphire-Q_dot_empty))
            
        return T_samp, cp_samp
