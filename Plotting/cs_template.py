import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import torch as tn
import random
import warnings
import multiprocessing as mp
from functools import partial
from scipy.integrate import quad, IntegrationWarning
import time
from datetime import date, time, datetime
from concurrent.futures import ThreadPoolExecutor
import matplotlib.colors as mcolors


# Import files
from constants import *
from rotation import*
from functions import  d_sigma, d_sigma_sm, sigma_sm, sme, sigma_full, summation_terms, integrate_sigma_hat_prime_sm, integrate_sigma_hat_prime_sme, dsigma_dQ, dsigma_dQ_1,dsigma_dQ_2,dsigma_dQ_3


#Don't foregt the metric convenction (+, -, -, -)
g = tn.tensor([
    [1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,-1]
], dtype=tn.float32)
CL1 = tn.tensor([
    [0, 0, 0, 0],
    [0, 1e-4, 0, 0],
    [0, 0, -1e-4, 0],
    [0,0, 0, 0]
], dtype=tn.float32)
CL2 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, -1e-4, 0],
    [0, -1e-4, 0, 0],
    [0,0, 0, 0]
], dtype=tn.float32)
CL3 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, -1e-4],
    [0, 0, 0, 0],
    [0,-1e-4, 0, 0]
], dtype=tn.float32)
CL4 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, -1e-4],
    [0,0,-1e-4, 0]
], dtype=tn.float32)

CLzz = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0,0,0, -1e-4]
], dtype=tn.float32)

CR = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=tn.float32)




p1 =  0.5*tn.tensor([1, 0, 0, 1], dtype=tn.float32)
p2 =  0.5*tn.tensor([1, 0, 0, -1], dtype=tn.float32)




### not quite sure what these are, potentially the wilson coefficient things?
quark_couplings1 = [(2, 0.2018666667, -0.046673592, 0.453326408)]
quark_couplings2 = [(1, -0.1009333333, 0.023336796, -0.476663204)]

def quark_coupling(q_list):
    #input: a list of the quark types desired as letters. some subset of:
    # ['u', 'd', 'c', 's', 'b', 't']
    
    quarks = [
    (2, 2/3*e, 'u', 1/2),
     (1, -1/3*e, 'd', -1/2),
     (3, -1/3*e, 's', -1/2),
     (4, 2/3*e, 'c', 1/2),
      (5, -1/3*e, 'b', -1/2),
     (6, 2/3*e, 't', 1/2),
    ]
    quarks = [i for i in quarks if i[2] in q_list]

    # List of quark properties and couplings
    quark_couplings = []

    for flavor, e_f, _, I3 in quarks:
        g_fR = -e_f * sin2th_w
        g_fL = I3 - e_f * sin2th_w
        
        # Rounding to 4 decimal places
        e_f = round(e_f, 10)
        g_fR = round(g_fR, 10)
        g_fL = round(g_fL, 10)
        
        quark_couplings.append((flavor, e_f, g_fR, g_fL))
    return quark_couplings



def rotation_matricies(start_time, end_time):
    #expects an astropy datetime object for botht he start and end time
    step_seconds = int(timedelta(hours=1).total_seconds())
    num_steps = (end_time - start_time) // step_seconds
    times = []
    contrelep1 = []
    contrelep2 = []

    R_y_lat = R_y(latitude)
    R_z_azi = R_z(azimuth)
    mat_cons = tn.matmul(R_y_lat,R_z_azi)
    # Main loop
    current_time = start_time
    for _ in range(num_steps):
        # Convert current_time to a timestamp
        current_datetime = datetime.fromtimestamp(current_time)
        time_utc = current_datetime.timestamp()

        # Calculate omega_t
        omega_t_sid = omega_utc * time_utc + 3.2830 
        # Construct the complete rotation matrix from SCF to CMS
        R_Z_omega = R_Z(omega_t_sid)
        R_mat = tn.matmul(R_Z_omega, mat_cons)
        R_matrix1 = tn.einsum('ma,an->mn', g, R_mat)
        R_matrix2 = tn.einsum('am,na->mn', g, R_mat)
        # print(R_matrix1)
        # Compute contrL and contrR using matrix multiplication
        contrp1 = tn.einsum('ij,j->i', R_matrix1, p1)
        contrp2 =  tn.einsum('ij,i->j',R_matrix2, p2)
        # Record the times and contr matrix elements
        times.append(current_time)
        contrelep1.append(contrp1)
        contrelep2.append(contrp2)


        # Move to the next time step
        current_time += step_seconds
    return contrelep1, contrelep2, times ### don't understand the theory well enough to understand why there are only two of these

def compute_result(args, sigma_sm_value, Q_min, Q_max):
    pm, pn, quark_couplings, CL1, CL2, CL3, CL4, CR = args
    
    # Compute the SME contributions
    result_sme1 = sme(Q_min, Q_max, CL1, CR, pm, pn, quark_couplings, sigma_sm_value)
    result_sme2 = sme(Q_min, Q_max, CL2, CR, pm, pn, quark_couplings, sigma_sm_value)
    result_sme3 = sme(Q_min, Q_max, CL3, CR, pm, pn, quark_couplings, sigma_sm_value)
    result_sme4 = sme(Q_min, Q_max, CL4, CR, pm, pn, quark_couplings, sigma_sm_value)
    
    # Add the SM result to each of the SME results after the loop
    final_result_sme1 = result_sme1 + sigma_sm_value
    final_result_sme2 = result_sme2 + sigma_sm_value
    final_result_sme3 = result_sme3 + sigma_sm_value
    final_result_sme4 = result_sme4 + sigma_sm_value
    
    # Return the result as a dictionary
    return {
        'result_sme1': final_result_sme1,
        'result_sme2': final_result_sme2,
        'result_sme3': final_result_sme3,
        'result_sme4': final_result_sme4
    }
    
def drell_yan_cs(Q_min, Q_max, times, quark_couplings, contrelep1, contrelep2):
    warnings.simplefilter("ignore", IntegrationWarning)

    sigma_sm_value = sigma_sm(Q_min, Q_max, quark_couplings)
    args_list = [(pm, pn, quark_couplings, CL1, CL2, CL3, CL4, CR) 
                for (pm, pn) in zip(contrelep1, contrelep2)]
    
    
    # Create a partial function to include sigma_sm_value
    partial_compute_result = partial(compute_result, sigma_sm_value=sigma_sm_value, Q_min = Q_min, Q_max = Q_max)

    # Create a multiprocessing Pool
    # with mp.Pool(mp.cpu_count()) as pool:
    #     # Pass the partial function to pool.map
    #     results = pool.map(partial_compute_result, args_list)

    results = [partial_compute_result(args) for args in args_list]

    # Function to convert timestamps to hours
    def convto_hours(timestamps):
        start_time = timestamps[0]  # The start time to normalize
        return [(t - start_time) / 3600 for t in timestamps]  # Convert seconds to hours

    # Perform conversion
    hours_start = convto_hours(times)
    dratios = [np.array([result[f'result_sme{i+1}'] / sigma_sm_value for result in results]) for i in range(4)]
    hours_array = np.array(hours_start)
    return dratios, hours_array
    
    

def compute_sme_for_bin(Q_range):
    # Unpack the tuple
    Q_start, Q_end = Q_range
    # Function to compute SME for a given range
    return sme(Q_start, Q_end, CLzz, CLzz, p1, p2, quark_couplings1)



def main_loop(quarks, time, time_delta, Q_min, Q_max, plot = False):
    ## time: [year, month (?), day, h?, m???]
    #### assumes timedelta is in days
    quark_couplings = quark_coupling(quarks)
    
    specific_time = datetime(time[0], time[1], time[2], time[3], time[4])
    start_time = int(specific_time.timestamp())
    end_time = start_time + int(timedelta(days=time_delta).total_seconds())

    contrelep1, contrelep2, times = rotation_matricies(start_time, end_time)
    dratios, hours_array = drell_yan_cs(Q_min, Q_max, times, quark_couplings, contrelep1, contrelep2)

    plt.figure(figsize=(10, 8))
    colors = ['mediumblue', 'red', 'goldenrod','limegreen' ]
    # Increase line width and adjust line styles for differentiation
    line_styles = [(5, (10, 3)), '--', '-.', '-']
    labels=['$c^{11}=-c^{22}=10^{-4}$','$c^{12}=c^{21}=10^{-4}$','$c^{13}=c^{31}=10^{-4}$','$c^{23}=c^{32}=10^{-4}$']

    for i in range(4):
        plt.step(hours_array, dratios[i], where='post', color=colors[i], label=labels[i], linewidth=2.5, linestyle=line_styles[i])

    # Customizing the legend: move it inside the plot area, adjust font size, and add a background
    plt.legend(loc='best', fontsize=12, frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray')

    # Adding labels and title with increased font size for clarity
    plt.xlabel('Time (hours)', fontsize=14)
    plt.ylabel(r'$\sigma_{SME}/\sigma_{SM}$', fontsize=14)
    plt.title(r'$SME/SM \; at \; Q \in [70,80] \; GeV$', fontsize=18, loc='left')

    # Add grid lines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.3)

    # Adjust tick parameters for better readability
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)  
    plt.tick_params(which='both', width=1.5)
    plt.tick_params(which='major', length=7, labelsize=12)
    plt.tick_params(which='minor', length=4, color='gray')
    plt.tick_params(axis='y', direction='in', which='both', labelsize=12) 

    # Customize x-ticks
    plt.xticks(ticks=range(0, 24, 1), labels=[str(hour) for hour in range(0, 24, 1)])

    # Adjust layout to prevent clipping of labels and title
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    # Save and show the plot
    plt.savefig("liv.png", bbox_inches='tight', pad_inches=0.1, dpi=300)
    plt.show()
    
main_loop(['u', 'd'], [2017, 1, 1, 0, 0], 1, 70, 80, True)
print("success")