#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:18:55 2026

Flow Regime Diagnostic Plots - Log-log plot
StreamLit slope fitting


@author: luiszerpa
"""


import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def logarithmicDerivative(time,pressure,log_cycle_fraction):
    '''Calculates the Bourdet pressure derivative using a three-point method.

    Args:
        time (np.array): Array of time values.
        pressure (np.array): Array of pressure values corresponding to the time values.
        log_cycle_fraction (float): The fraction of a log cycle (DX) used to 
                                    select points for differentiation. Typical 
                                    values range from 0.01 to 0.2.

    Returns:
        np.array: Array of calculated derivatives.'''
    
    n = len(time)
    derivative = np.zeros(n)
    
    # Ensure time is sorted and positive
    if not np.all(np.diff(time) > 0):
        raise ValueError("Time array must be monotonically increasing.")
    if not np.all(time >= 0):
        raise ValueError("Time values must be positive for logarithmic analysis.")

    # Iterate through each point to calculate its derivative
    for i in range(n):
        # Find points 'before' (j) and 'after' (k) the current point (i)
        # based on the specified log_cycle_fraction (DX).
        # We look for points that satisfy: 
        # log10(time[i] / time[j]) >= DX and log10(time[k] / time[i]) >= DX
        
        # Search backward for point j
        j = i - 1
        while j >= 0 and np.log10(time[i] / time[j]) < log_cycle_fraction:
            j -= 1
        
        # Search forward for point k
        k = i + 1
        while k < n and np.log10(time[k] / time[i]) < log_cycle_fraction:
            k += 1

        if j >= 0 and k < n:
            # Use the three-point formula if valid j and k are found
            # The formula is essentially a weighted average of slopes:
            # dp/d(ln t) = [ (p_R - p_C)/(ln(t_R / t_C)) * ln(t_C / t_L) + (p_C - p_L)/(ln(t_C / t_L)) * ln(t_R / t_C) ] / ln(t_R / t_L)
            
            ln_ti_tj = np.log(time[i] / time[j])
            ln_tk_ti = np.log(time[k] / time[i])
            ln_tk_tj = np.log(time[k] / time[j])
            
            # The derivative formula as found in the Bourdet literature
            derivative[i] = ((pressure[k] - pressure[i])/(ln_tk_ti) * ln_ti_tj + 
                             (pressure[i] - pressure[j])/(ln_ti_tj) * ln_tk_ti) / ln_tk_tj
        else:
            # For points at the boundaries, the derivative cannot be reliably 
            # calculated with this method, so we set it to NaN or a default.
            derivative[i] = np.nan
            
    return derivative

st.title("Log-log Plot - Slope Fiting")


time_data = np.array([0.005,0.0106,0.017,0.0241,0.0321,0.0411,0.0512,0.0626,0.0755,0.0899,0.1061,0.1244,0.1449,0.1681,0.1941,0.223,0.256,0.293,0.335,0.382,0.435,0.494,0.561,0.636,0.72,0.815,0.922,1.042,1.178,1.33,1.501,1.694,1.91,2.154,2.428,2.737,3.084,3.474,3.914,4.408,4.964,5.589,6.293,7.085,7.975,8.977,10.1,11.37,12.8,14.4,16.21,18.24,20.53,23.1,25.99,29.24,32.9,37.02,41.65,46.86,52.73,59.32,66.74,72])
pressure_data = np.array([1915.78,1902.13,1888.64,1875.96,1864.29,1853.57,1843.77,1835.07,1827.67,1821.33,1815.91,1811.71,1808.05,1805.05,1802.93,1800.97,1799.43,1797.77,1796.77,1795.37,1794.52,1793.3,1792.26,1791.22,1790.33,1789.49,1788.81,1787.79,1786.91,1786.03,1785.14,1784.61,1783.64,1782.78,1782.06,1781.18,1780.65,1779.54,1778.93,1778.21,1777.36,1776.6,1775.71,1774.85,1774.08,1773.25,1772.31,1771.28,1770.46,1769.4,1768.32,1767.14,1765.98,1764.75,1763.21,1761.86,1760.29,1758.76,1757.2,1755.29,1753.34,1751.19,1749.15,1747.51])
p_i = 1930 # psia
DeltaP = p_i - pressure_data

# Calculate the derivative using a log cycle fraction of 0.1
derivs = logarithmicDerivative(time_data, DeltaP, log_cycle_fraction=0.1)


x0 = np.zeros((4))
x0[0] = st.slider("Horizontal line", 0.0, 15.0, 1.0)
x0[1] = st.slider("1/2 Slope", 0.0, 15.0, 1.0)
x0[2] = st.slider("1/4 Slope", 0.0, 15.0, 1.0)
x0[3] = st.slider("-1/2 Slope", 0.0, 15.0, 1.0)


nZerob = x0[0]
nZeroSlope = nZerob*np.ones(np.size(time_data))

nHalfb = x0[1]
nHalfSlope = nHalfb*time_data**0.5

nQuarterb = x0[2]
nQuarterSlope = nQuarterb*time_data**0.25

nNegHalfb = x0[3]
nNegHalfSlope = nNegHalfb*time_data**(-0.5)

      
# Log-log plot
fig, ax = plt.subplots(figsize=(8, 5))  # Create a figure containing a single axes.
ax.plot(time_data,DeltaP,'ob',label='Pressure') # Plot data on the axes.
ax.plot(time_data, derivs, 'xr', label='Log Derivative')
ax.plot(time_data, nZeroSlope, '--r', linewidth=2, label='Horizontal line')
ax.plot(time_data, nHalfSlope, '--b', linewidth=2, label='1/2 slope')
ax.plot(time_data, nQuarterSlope, '--g', linewidth=2, label='1/4 slope')
ax.plot(time_data, nNegHalfSlope, '--y', linewidth=2, label='-1/2 slope')
ax.set_xlabel('Time (hr)',fontsize='12',fontweight='bold', fontname='Verdana')
ax.set_ylabel('$\Delta$Pressure (psia)',fontsize='12',fontweight='bold',fontname='Verdana')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Log-Log Plot")
ax.tick_params(direction='in',width=1.5,length=8.0, labelsize=12.0)
# ax.set(xlim=(0.001, 1000), ylim=(0.001, 1000))
for tick in ax.get_xticklabels():
    tick.set_fontname('Verdana')
for tick in ax.get_yticklabels():
    tick.set_fontname('Verdana')
ax.grid(True)  
