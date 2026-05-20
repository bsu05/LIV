import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
from scipy.optimize import minimize_scalar
import pdb
import time

# Import files
from constants import *
from rotation import*
from functions import  d_sigma, d_sigma_sm, sigma_sm, sme, sigma_full, summation_terms, integrate_sigma_hat_prime_sm, integrate_sigma_hat_prime_sme, dsigma_dQ, dsigma_dQ_1,dsigma_dQ_2,dsigma_dQ_3
from multiprocessing import Pool
import pandas as pd

#Don't foregt the metric convenction (+, -, -, -)
g = tn.tensor([
    [1,0,0,0],
    [0,-1,0,0],
    [0,0,-1,0],
    [0,0,0,-1]
], dtype=tn.float32)
Cxx = tn.tensor([
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, -1, 0],
    [0,0, 0, 0]
], dtype=tn.float32)
Cxy = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, -1, 0],
    [0, -1, 0, 0],
    [0,0, 0, 0]
], dtype=tn.float32)
Cxz = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, -1],
    [0, 0, 0, 0],
    [0,-1, 0, 0]
], dtype=tn.float32)
Cyz = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, -1],
    [0,0,-1, 0]
], dtype=tn.float32)

CLzz = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0,0,0, -1]
], dtype=tn.float32)

C0 = tn.tensor([
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
], dtype=tn.float32)



### these are the two momenta
p1 =  0.5*tn.tensor([1, 0, 0, 1], dtype=tn.float32)
p2 =  0.5*tn.tensor([1, 0, 0, -1], dtype=tn.float32)


### not quite sure what these are
quark_couplings1 = [(2, 0.2018666667, -0.046673592, 0.453326408)]
quark_couplings2 = [(1, -0.1009333333, 0.023336796, -0.476663204)]


quarks_ref = [
    (2, 2/3*e, 'u', 1/2),
    (1, -1/3*e, 'd', -1/2),
    (3, -1/3*e, 's', -1/2),
    (4, 2/3*e, 'c', 1/2),
    (5, -1/3*e, 'b', -1/2),
    (6, 2/3*e, 't', 1/2),
]
    
def quark_coupling(q_list): ## no multithreading
    #input: a list of the quark types desired as letters. some subset of:
    # ['u', 'd', 'c', 's', 'b', 't']
    quarks = [i for i in quarks_ref if i[2] in q_list]

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

def rotation_matricies(start_time, end_time): ### no multithreading
    #expects an astropy datetime object for both the start and end time
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

def compute_result(args, sigma_sm_value, Q_min, Q_max, CL1, CL2, CL3, CL4, CR): ## no multithreading
    pm, pn, quark_couplings, CL1, CL2, CL3, CL4, CR = args
    
    # Compute the SME contributions
    ### im going to want to change these to allow for different things to be calculated  
    ### sme uses simpsons rule but this is not multithreaded (should it be?)
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
    
def compute_single_result(args, sigma_sm_value, Q_min, Q_max, CL, CR): ## not multithreaded
    pm, pn, quark_couplings, CL, CR = args
    
    # Compute the SME contributions
    ### im going to want to change these to allow for different things to be calculated  
    result = sme(Q_min, Q_max, CL, CR, pm, pn, quark_couplings, sigma_sm_value)
    final_result = result + sigma_sm_value
    return {
        'result_sme': final_result,
    }
    
def drell_yan_cs_single(Q_min, Q_max, times, quark_couplings, contrelep1, contrelep2, CL, CR): ## multithreaded
    warnings.simplefilter("ignore", IntegrationWarning) 

    sigma_sm_value = sigma_sm(Q_min, Q_max, quark_couplings)
    args_list = [(pm, pn, quark_couplings, CL, CR) 
                for (pm, pn) in zip(contrelep1, contrelep2)]
    
    # Create a partial function to include sigma_sm_valu
    
    partial_compute_result = partial(compute_single_result, sigma_sm_value=sigma_sm_value, Q_min = Q_min, Q_max = Q_max, CL=CL, CR=CR)
    # Create a multiprocessing Pool
    with mp.Pool(mp.cpu_count()) as pool:
        # Pass the partial function to pool.map 
        results = pool.map(partial_compute_result, args_list)


    # Function to convert timestamps to hours
    def convto_hours(timestamps):
        start_time = timestamps[0]  # The start time to normalize
        return [(t - start_time) / 3600 for t in timestamps]  # Convert seconds to hours

    # Perform conversion
    hours_start = convto_hours(times)
    dratios = [np.array([result['result_sme'] / sigma_sm_value for result in results])]
    hours_array = np.array(hours_start)
    return dratios, hours_array
    
def drell_yan_cs(Q_min, Q_max, times, quark_couplings, contrelep1, contrelep2, CL1, CL2, CL3, CL4, CR):
    warnings.simplefilter("ignore", IntegrationWarning)

    sigma_sm_value = sigma_sm(Q_min, Q_max, quark_couplings)
    args_list = [(pm, pn, quark_couplings, CL1, CL2, CL3, CL4, CR) 
                for (pm, pn) in zip(contrelep1, contrelep2)]
    
    
    # Create a partial function to include sigma_sm_value
    partial_compute_result = partial(compute_result, sigma_sm_value=sigma_sm_value, Q_min = Q_min, Q_max = Q_max, CL1=CL1, CL2=CL2, CL3=CL3, CL4=CL4, CR=CR)

    # Create a multiprocessing Pool
    with mp.Pool(mp.cpu_count()) as pool:
        # Pass the partial function to pool.map
        results = pool.map(partial_compute_result, args_list)

    # Function to convert timestamps to hours
    def convto_hours(timestamps):
        start_time = timestamps[0]  # The start time to normalize
        return [(t - start_time) / 3600 for t in timestamps]  # Convert seconds to hours

    # Perform conversion
    hours_start = convto_hours(times)
    dratios = [np.array([result[f'result_sme{i+1}'] / sigma_sm_value for result in results]) for i in range(4)]
    hours_array = np.array(hours_start)
    return dratios, hours_array

def calculate_variations(quarks, time, time_delta, Q_min, Q_max, CL_coeffs, CR_coeffs, which_tensor, single = False):
    ## time: [year, month (?), day, h?, m???]
    #### assumes timedelta is in days

    quark_couplings = quark_coupling(quarks)

    specific_time = datetime(time[0], time[1], time[2], time[3], time[4])
    start_time = int(specific_time.timestamp())
    end_time = start_time + int(timedelta(days=time_delta).total_seconds())

    contrelep1, contrelep2, times = rotation_matricies(start_time, end_time)
    
    tensors = [C0, Cxx, Cxy, Cxz, Cyz]

    if single:
        dratios, hours_array = drell_yan_cs_single(Q_min, Q_max, times, quark_couplings, contrelep1, contrelep2, tensors[which_tensor[0]]*CL_coeffs[0], tensors[which_tensor[1]]*CR_coeffs[0])
    else:
        if len(which_tensor[0]) == 1:
            #### this is actually not implemented yet, need to allow for the right handed one to vary
            CL1 = tensors[which_tensor[1, 0]]*CR_coeffs[0]
            CL2 = tensors[which_tensor[1, 1]]*CR_coeffs[1]
            CL3 = tensors[which_tensor[1, 2]]*CR_coeffs[2]
            CL4 = tensors[which_tensor[1, 3]]*CR_coeffs[3]
            dratios, hours_array = drell_yan_cs(Q_min, Q_max, times, quark_couplings, contrelep1, contrelep2, CL1, CL2, CL3, CL4, tensors[4])
        elif len(which_tensor[1]) == 1:
            CL1 = tensors[which_tensor[0, 0]]*CL_coeffs[0]
            CL2 = tensors[which_tensor[0, 1]]*CL_coeffs[1]
            CL3 = tensors[which_tensor[0, 2]]*CL_coeffs[2]
            CL4 = tensors[which_tensor[0, 3]]*CL_coeffs[3]
            dratios, hours_array = drell_yan_cs(Q_min, Q_max, times, quark_couplings, contrelep1, contrelep2, CL1, CL2, CL3, CL4, tensors[4])
    return dratios, hours_array



def make_variation_plot(dratios, hours_array, cl_coeffs, outname, cr_coeffs = [], yrange = [0.95, 1.05], coeff_type = 'L', bin_low = 70, bin_high = 80):
    #### need to make this more adaptable
    yrange = [0.95, 1.05]
    plt.figure(figsize=(10, 8))
    colors = ['mediumblue', 'red', 'goldenrod','limegreen' ]
    # Increase line width and adjust line styles for differentiation
    line_styles = [(5, (10, 3)), '--', '-.', '-']
    if coeff_type == 'both':
        cl_label = "{:.2e}".format(cl_coeffs[0])
        cr_label = "{:.2e}".format(cr_coeffs[0])
        labels=['$c_L^{11}=-c_L^{22}=$' + f'{cl_label}', '$c_R^{11}=-c_R^{22}=$' + f'{cr_label}']

        for i in range(len(dratios)):
            plt.step(hours_array, np.array(dratios[i][0]), where='post', color=colors[i], label=labels[i], linewidth=2.5, linestyle=line_styles[i])
            
    elif coeff_type == 'all':
        cl_up_label = "{:.2e}".format(cl_coeffs[0])
        cr_up_label = "{:.2e}".format(cr_coeffs[0])
        cl_down_label = "{:.2e}".format(cl_coeffs[0])
        cr_down_label = "{:.2e}".format(cr_coeffs[0])
        labels=['$c_{u,L}^{11}=$' + f'{cl_up_label}', '$c_{u,R}^{11}=$' + f'{cr_up_label}', '$c_{d,L}^{11}=$' + f'{cl_down_label}', '$c_{d,R}^{11}=$' + f'{cr_down_label}']

        for i in range(len(dratios)):
            plt.step(hours_array, np.array(dratios[i][0]), where='post', color=colors[i], label=labels[i], linewidth=2.5, linestyle=line_styles[i])
        
    else:
        c1_label = "{:.2e}".format(cl_coeffs[0])
        labels=['$c_R^{11}=-c_R^{22}=$' + f'{c1_label}']
        if len(cl_coeffs) > 1:
            c2_label = "{:.2e}".format(cl_coeffs[1])
            c3_label = "{:.2e}".format(cl_coeffs[2])
            c4_label = "{:.2e}".format(cl_coeffs[3])
            labels=['$c_L^{11}=-c_L^{22}=$' + f'{c1_label}','$c_L^{12}=c_L^{21}=$'+ f'{c2_label}', '$c_L^{13}=c_L^{31}=$'+f'{c3_label}','$c_L^{23}=c_L^{32}=$' +f'{c4_label}']
            
            if coeff_type != 'L':
                labels=['$c_R^{11}=-c_R^{22}=$' + f'{c1_label}','$c_R^{12}=c_R^{21}=$'+ f'{c2_label}', '$c_R^{13}=c_R^{31}=$'+f'{c3_label}','$c_R^{23}=c_R^{32}=$' +f'{c4_label}']
        for i in range(len(cl_coeffs)):
            plt.step(hours_array, dratios[i], where='post', color=colors[i], label=labels[i], linewidth=2.5, linestyle=line_styles[i])

    # Customizing the legend: move it inside the plot area, adjust font size, and add a background
    plt.legend(loc='best', fontsize=12, frameon=True, fancybox=True, framealpha=0.8, edgecolor='gray')

    # Adding labels and title with increased font size for clarity
    plt.xlabel('Time (hours)', fontsize=14)
    plt.ylabel(r'$\sigma_{SME}/\sigma_{SM}$', fontsize=14)
    plt.title(r'$SME/SM \; at \; Q \in $' + f'[{bin_low},{bin_high}]' + r'$\; GeV$', fontsize=18, loc='left')

    # Add grid lines for better readability
    plt.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.3)

    # Adjust tick parameters for better readability
    plt.minorticks_on()
    plt.tick_params(axis='x', which='minor', bottom=False)  
    plt.tick_params(which='both', width=1.5)
    plt.tick_params(which='major', length=7, labelsize=12)
    plt.tick_params(which='minor', length=4, color='gray')
    plt.tick_params(axis='y', direction='in', which='both', labelsize=12) 
    plt.ylim(yrange)

    # Customize x-ticks
    plt.xticks(ticks=range(0, 24, 1), labels=[str(hour) for hour in range(0, 24, 1)])

    # Adjust layout to prevent clipping of labels and title
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])

    # Save and show the plot
    plt.savefig(f"{outname}.png", bbox_inches='tight', pad_inches=0.1, dpi=300)


    
def compute_sme_for_bin(Q_range):
    # Unpack the tuple
    Q_start, Q_end, CLzz_coeff, flavor = Q_range
    # Function to compute SME for a given range
    qc = quark_coupling(flavor)
    return sme(Q_start, Q_end, CLzz*CLzz_coeff, CLzz*CLzz_coeff, p1, p2, quark_couplings1)

def compute_mass_bins(Q_bins, CLzz_coeff, flavor):
    # Prepare the list of arguments for each bin
    bin_ranges = [(Q_bins[i], Q_bins[i + 1], CLzz_coeff, flavor) for i in range(len(Q_bins) - 1)]

    # Use multiprocessing to compute SME values
    with mp.Pool() as pool:
        sme_values2 = pool.map(compute_sme_for_bin, bin_ranges)
    return sme_values2

def make_mass_bin_plot(sme_vals, Q_bins,  filename):
    plt.figure(figsize=(8, 6))
    plt.step(Q_bins,  sme_vals + [sme_vals[-1]], where='post', color='blue', label = '$c^{33}_{d}=10^{-4}$') ### need to change this
    
    plt.xlabel('Q [GeV]')
    plt.yscale('log')
    plt.ylabel('$\\sigma_{LV} \\;[Pb]$')
    # plt.title('Lorentz violation contribution in the cross section')
    plt.grid(True, which="both", ls="--")  

    plt.tick_params(axis='both', which='both', direction='in', top=True, right=True)

    plt.tight_layout()
    plt.legend(fontsize=11, loc='best')
    # Save and show the plot
    plt.savefig(f'{filename}.png', dpi=300)
    plt.show()


def find_closest_input(func, target, bounds=(-1000, 1000), tolerance = 5e-5, max_iter = 10):
    def objective(x):
        return (func(x) - target)**2
    
    best_x = None
    best_diff = float('inf')

    for i in range(max_iter):
        result = minimize_scalar(objective, bounds=bounds, method='bounded')
        if not result.success:
            break

        x = result.x
        diff = abs(func(x) - target)

        if diff < best_diff:
            best_diff = diff
            best_x = x

        print(best_diff)
        # Early stop if within tolerance
        if best_diff <= tolerance:
            break

        # Optional: shrink bounds around the current best_x for refinement
        range_shrink = (bounds[1] - bounds[0]) * 0.5**(i+1)
        bounds = (max(bounds[0], best_x - range_shrink), min(bounds[1], best_x + range_shrink))
    return best_x


def calculate_amplitudes():
    fileout = "/work/submit/jbenke/WRemnants/scripts/corrections/quark_liv_scalings.npy"
    cL_coeffs = [1e-4]

    liv_amp = np.zeros([len(Q_range) - 1, 4, 2, 24])
    # pdb.set_trace()
    for i in range(len(Q_range) - 1):
        low = Q_range[i]
        high = Q_range[i+1]
        for j in range(0, 4):
            dratios, hours_array = calculate_variations(['u'], [2016, 1, 1, 0, 0], 1, low, high, cL_coeffs, single = True, which_tensor = j)
            # pdb.set_trace()
            liv_amp[i, j, 0, :] = dratios[0]
            print(dratios[0])
            
            dratios, hours_array = calculate_variations(['d'], [2016, 1, 1, 0, 0], 1, low, high, cL_coeffs, single = True, which_tensor = j)
            liv_amp[i,j, 1, :] = dratios[0]
            print(dratios[0])
            print("-----------")
    np.save(fileout, liv_amp)




#quarks, time, time_delta, Q_min, Q_max, plot = False
### oof so the amplitude of this variation is also mass dependent
# cLzz_coeff = 1e-4

Q_range = [50, 60.3, 85.2298, 88.1398, 89.3644, 90.16, 90.8102, 91.428, 92.1163, 93.0461, 94.9463, 120, 130]
## structure is mass bin, coeff #, u/d, value

#### UP VALUES
cL_coeffs = [2.5e-4,2.5e-4,1.2e-4,1.2e-4]
### UP UNCERTAINTIES
cL_coeffs = [2.5e-4, 2.5e-4, 1.2e-4, 1.2e-4] ### values are too large
cL_coeffs = [0]
cR_coeffs = [1e-4]
start = time.time()


# calculate_variations(['u'], [2016, 1, 1, 0, 0], 1, 50, 60.3 , cL_coeffs, cR_coeffs, [0, 1], True)
args = [['u'], [2016, 1, 1, 0, 0], 1, 50, 60.3 , cL_coeffs, cR_coeffs, [0, 1], True]

with mp.Pool(mp.cpu_count()) as pool:
    results = calculate_variations(*args)

end = time.time()
print(f"time: {end - start}")




# dratios_r_up, hours_array_r = calculate_variations(['u'], [2016, 1, 1, 0, 0], 1, 50, 60.3 , cL_coeffs, cR_coeffs, [0, 1], single = True)
# dratios_l_up, hours_array = calculate_variations(['u'], [2016, 1, 1, 0, 0], 1, 50, 60.3 , cR_coeffs, cL_coeffs, [1, 0], single = True)

# dratios_r_down, hours_array_r = calculate_variations(['d'], [2016, 1, 1, 0, 0], 1, 50, 60.3 , cL_coeffs, cR_coeffs, [0, 1], single = True)
# dratios_l_down, hours_array = calculate_variations(['d'], [2016, 1, 1, 0, 0], 1, 50, 60.3 , cR_coeffs, cL_coeffs, [1, 0], single = True)
# # print("variation successful")
# file_out = '/home/submit/jbenke/public_html/'
# dratios = [dratios_l_up, dratios_r_up, dratios_l_down, dratios_r_down]

# make_variation_plot(dratios, hours_array, cR_coeffs, f'{file_out}_comparison', cr_coeffs=cR_coeffs, coeff_type='all', bin_low = 50, bin_high = 60.3)


''' 
coeff = [1e-4]
quarks = ['u', 'd']
Q_range = np.linspace(15, 120, 15) 
## structure is mass bin, coeff #, u/d, l/r, value
amplitudes = np.zeros([len(Q_range)-1, 4, 2, 2, 24])
amp_max = np.copy(amplitudes)
for i in range(len(Q_range)-1):
    print(f"mass bin: {i}")
    for j in range(1, 5):
        print(f"coefficient: {j}")
        for k in range(len(quarks)) :
            bin_low = Q_range[i]
            bin_high = Q_range[i+1]
            dratios_l, _ = calculate_variations([quarks[k]], [2016, 1, 1, 0, 0], 1, bin_low, bin_high, coeff, [0], [j, 0], single = True)
            dratios_r, hours_array = calculate_variations([quarks[k]], [2016, 1, 1, 0, 0], 1, bin_low, bin_high, [0], coeff, [0, j], single = True)

            amplitudes[i, j-1, k, 0] = dratios_l[0]
            amplitudes[i, j-1, k, 1] = dratios_r[0]
            amp_max[i, j-1, k, 0] = np.max(dratios_l[0])
            amp_max[i, j-1, k, 1] = np.max(dratios_r[0])
            
            
np.save("/work/submit/jbenke/WRemnants/scripts/corrections/liv_amplitudes_lr_FINE.npy", amplitudes)
np.save("/work/submit/jbenke/WRemnants/scripts/corrections/liv_max_amplitudes_lr_FINE.npy", amp_max)

'''






'''
#### DOWN VALUES
# cL_coeffs = [2.5e-4,2.5e-4,1.2e-4,1.2e-4]

### DOWN UNCERTAINTIES
# cL_coeffs = [2e-4, 2e-4, 2e-4, 1.5e-2] ### values are too large
dratios, hours_array = calculate_variations(['u'], [2016, 1, 1, 0, 0], 1, 50, 60.3 , cL_coeffs)
make_variation_plot(dratios, hours_array, cL_coeffs, f'{file_out}down_quark')

# Q_bins = np.array([50, 60.3, 85.2298, 88.1398, 89.3644, 90.16, 90.8102, 91.428, 92.1163, 93.0461, 94.9463, 120, 130])
 # sme_vals = compute_mass_bins(Q_bins, cLzz_coeff, flavor)
# make_mass_bin_plot(sme_vals, Q_bins, f'{file_out}mass_bins')



# Example usage
def nonlinear_function(cl_coeff):
    dratios, _ =  calculate_variations(['u', 'd'], [2016, 1, 1, 0, 0], 1, 70, 80, [cl_coeff], single = True)
    print(np.max(dratios))
    return np.max(dratios)
 

# target_value = [1.002321, 1.002328, 1.0023273, 1.0023204]
# for val in target_value:
#     closest_input = find_closest_input(nonlinear_function, val, bounds=(1e-6, 1e-3))
#     print("final_answer")
#     print(closest_input)
'''


