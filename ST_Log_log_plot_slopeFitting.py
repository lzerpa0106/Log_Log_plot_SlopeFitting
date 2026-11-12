#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 11:18:55 2026

Flow Regime Diagnostic Plots - Log-log plot
StreamLit slope fitting


@author: luiszerpa
"""


import io

import numpy as np
import pandas as pd
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

def load_csv(uploaded_file):
    '''Reads a time/pressure CSV and returns (DataFrame, p_i or None).

    Expected format (header row required, '#' lines are comments):
        # p_i = 1930            <- optional: initial pressure, psia
        time_hr,pressure_psia
        0.005,1915.78
        ...
    Column names are flexible; the user picks them in the app.
    '''
    text = uploaded_file.getvalue().decode("utf-8-sig")
    p_i = None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") and "p_i" in line.lower() and "=" in line:
            try:
                p_i = float(line.split("=", 1)[1])
            except ValueError:
                pass
    df = pd.read_csv(io.StringIO(text), comment="#", skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    return df, p_i


def guess_column(columns, keywords, fallback):
    '''Returns the index of the first column whose name contains a keyword.'''
    for i, c in enumerate(columns):
        if any(k in c.lower() for k in keywords):
            return i
    return fallback


def clean_data(time, pressure):
    '''Drops NaN/non-positive times, sorts by time and removes duplicate times.'''
    mask = np.isfinite(time) & np.isfinite(pressure) & (time > 0)
    time, pressure = time[mask], pressure[mask]
    order = np.argsort(time, kind="stable")
    time, pressure = time[order], pressure[order]
    time, unique_idx = np.unique(time, return_index=True)
    return time, pressure[unique_idx]


st.title("Log-log Plot - Slope Fiting")

# ---------------------------------------------------------------------------
# Data source: built-in example or user CSV
# ---------------------------------------------------------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader(
    "Upload time/pressure CSV", type=["csv", "txt"],
    help="CSV with a header row, e.g. 'time_hr,pressure_psia'. "
         "Optional first line '# p_i = 1930' sets the initial pressure.")
log_cycle_fraction = st.sidebar.slider(
    "Derivative smoothing (log-cycle fraction L)", 0.01, 0.5, 0.1, 0.01)

csv_p_i = None
if uploaded is not None:
    try:
        df, csv_p_i = load_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")
        st.stop()
    if df.shape[1] < 2:
        st.error("The CSV needs at least two columns (time and pressure).")
        st.stop()
    cols = list(df.columns)
    t_col = st.sidebar.selectbox("Time column", cols,
                                 index=guess_column(cols, ["time", "hr", "t"], 0))
    p_col = st.sidebar.selectbox("Pressure column", cols,
                                 index=guess_column(cols, ["pres", "psi", "p"], 1))
    time_raw = pd.to_numeric(df[t_col], errors="coerce").to_numpy(dtype=float)
    pres_raw = pd.to_numeric(df[p_col], errors="coerce").to_numpy(dtype=float)
    time_data, pressure_data = clean_data(time_raw, pres_raw)
    dropped = len(time_raw) - len(time_data)
    st.sidebar.caption(f"{uploaded.name}: {len(time_data)} points used"
                       + (f", {dropped} dropped (blank/non-numeric/t≤0/duplicate)" if dropped else ""))
    if len(time_data) < 3:
        st.error("Need at least 3 valid points to compute the derivative.")
        st.stop()
else:
    st.sidebar.caption("Using built-in example data.")
    time_data = np.array([0.005,0.0106,0.017,0.0241,0.0321,0.0411,0.0512,0.0626,0.0755,0.0899,0.1061,0.1244,0.1449,0.1681,0.1941,0.223,0.256,0.293,0.335,0.382,0.435,0.494,0.561,0.636,0.72,0.815,0.922,1.042,1.178,1.33,1.501,1.694,1.91,2.154,2.428,2.737,3.084,3.474,3.914,4.408,4.964,5.589,6.293,7.085,7.975,8.977,10.1,11.37,12.8,14.4,16.21,18.24,20.53,23.1,25.99,29.24,32.9,37.02,41.65,46.86,52.73,59.32,66.74,72])
    pressure_data = np.array([1915.78,1902.13,1888.64,1875.96,1864.29,1853.57,1843.77,1835.07,1827.67,1821.33,1815.91,1811.71,1808.05,1805.05,1802.93,1800.97,1799.43,1797.77,1796.77,1795.37,1794.52,1793.3,1792.26,1791.22,1790.33,1789.49,1788.81,1787.79,1786.91,1786.03,1785.14,1784.61,1783.64,1782.78,1782.06,1781.18,1780.65,1779.54,1778.93,1778.21,1777.36,1776.6,1775.71,1774.85,1774.08,1773.25,1772.31,1771.28,1770.46,1769.4,1768.32,1767.14,1765.98,1764.75,1763.21,1761.86,1760.29,1758.76,1757.2,1755.29,1753.34,1751.19,1749.15,1747.51])
    csv_p_i = 1930.0  # psia

# Initial pressure: from the CSV '# p_i =' line if present, editable by the user
default_p_i = csv_p_i if csv_p_i is not None else float(np.max(pressure_data))
p_i = st.sidebar.number_input("Initial pressure p_i (psia)", value=float(default_p_i))
DeltaP = np.abs(p_i - pressure_data)   # abs() handles drawdown and buildup

# Calculate the Bourdet derivative
derivs = logarithmicDerivative(time_data, DeltaP, log_cycle_fraction=log_cycle_fraction)

with st.expander("Data table"):
    st.dataframe(pd.DataFrame({"time_hr": time_data, "pressure_psia": pressure_data,
                               "DeltaP_psi": DeltaP, "derivative_psi": derivs}))


x0 = np.zeros((5))
x0[0] = st.slider("Horizontal line", 0.0, 50.0, 1.0)
x0[1] = st.slider("Unit Slope", 0.0, 100.0, 1.0)
x0[2] = st.slider("1/2 Slope", 0.0, 50.0, 1.0)
x0[3] = st.slider("1/4 Slope", 0.0, 50.0, 1.0)
x0[4] = st.slider("-1/2 Slope", 0.0, 50.0, 1.0)


nZerob = x0[0]
nZeroSlope = nZerob*np.ones(np.size(time_data))

nUnitb = x0[1]
nUnitSlope = nUnitb*time_data

nHalfb = x0[2]
nHalfSlope = nHalfb*time_data**0.5

nQuarterb = x0[3]
nQuarterSlope = nQuarterb*time_data**0.25

nNegHalfb = x0[4]
nNegHalfSlope = nNegHalfb*time_data**(-0.5)

      
# Log-log plot
fig, ax = plt.subplots(figsize=(8, 5))  # Create a figure containing a single axes.
ax.plot(time_data,DeltaP,'ob',label='Pressure') # Plot data on the axes.
ax.plot(time_data, derivs, 'xr', label='Log Derivative')
ax.plot(time_data, nZeroSlope, '--r', linewidth=2, label='Horizontal line')
ax.plot(time_data, nUnitSlope, '--c', linewidth=2, label='Unit Slope')
ax.plot(time_data, nHalfSlope, '--b', linewidth=2, label='1/2 slope')
ax.plot(time_data, nQuarterSlope, '--g', linewidth=2, label='1/4 slope')
ax.plot(time_data, nNegHalfSlope, '--y', linewidth=2, label='-1/2 slope')
ax.set_xlabel('Time (hr)',fontsize='12',fontweight='bold', fontname='Verdana')
ax.set_ylabel(r'$\Delta$Pressure (psia)',fontsize='12',fontweight='bold',fontname='Verdana')
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
ax.legend()
st.pyplot(fig)
