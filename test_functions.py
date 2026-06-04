import numpy as np
import torch as tn
from scipy.integrate import quad
import lhapdf
from constants import *
from scipy.integrate import simps

pdf = lhapdf.mkPDF("NNPDF31_nnlo_as_0118", 0)

factor = 4 * alpha**2*np.pi / (3 * Nc)

def f_s(x, tau, flavor, Q2):
    tau_x = tau / x
    pdf_flavor_x = pdf.xfxQ2(flavor, x, Q2)
    pdf_flavor_tau_x = pdf.xfxQ2(flavor, tau_x, Q2)
    pdf_anti_flavor_x = pdf.xfxQ2(-flavor, x, Q2)
    pdf_anti_flavor_tau_x = pdf.xfxQ2(-flavor, tau_x, Q2)

    term1 = (1 / x) * pdf_flavor_x * (1/tau_x) * pdf_anti_flavor_tau_x
    term2 = (1/tau_x) * pdf_flavor_tau_x * (1 / x) * pdf_anti_flavor_x
    
    return term1 + term2


def num_derivative(func, x, h=1e-8, *args):
    return (func(x + h, *args) - func(x - h, *args)) / (2 * h)


def f_prime_s(x, tau, flavor, Q2):
    tau_x = tau/x
    f_f_tau_x_prime = num_derivative(lambda t: 1/t * pdf.xfxQ2(flavor, t, Q2), tau_x)
    f_fbar_tau_x_prime = num_derivative(lambda t: 1/t * pdf.xfxQ2(-flavor, t, Q2), tau_x)

    pdf_flavor_x = pdf.xfxQ2(flavor, x, Q2)
    pdf_anti_flavor_x = pdf.xfxQ2(-flavor, x, Q2)
    
    return (1/x * pdf_flavor_x * f_fbar_tau_x_prime + \
           1/x * f_f_tau_x_prime * pdf_anti_flavor_x)


def sigma_hat_prime(x, tau, C, p1, p2, flavor, Q2):
    tau_x = tau/x

    f_s_val = f_s(x, tau, flavor, Q2)
    f_prime_s_val = f_prime_s(x, tau, flavor, Q2)

    # Efficiently handle the contraction with non-zero elements of C
    contraction_p1p1 = tn.einsum('mn,m,n->', C, p1, p1)
    contraction_p1p2 = tn.einsum('mn,m,n->', C, p1, p2)
    contraction_p2p1 = tn.einsum('mn,m,n->', C, p2, p1)
    contraction_p2p2 = tn.einsum('mn,m,n->', C, p2, p2)
    
    term1 = f_s_val
    
    term2 = 2* (1 + x / tau_x) * (contraction_p1p1 + contraction_p1p2 +  contraction_p2p1 + contraction_p2p2) * f_s_val
    
    term3 = 2 * (x * contraction_p1p1 +  tau_x * contraction_p1p2 + tau_x * contraction_p2p1 + x * contraction_p2p2) * f_prime_s_val
    
    return term1, term2 + term3

def term_1(Q2, e_f):
    return e_f**2 / (2*Q2**2)
    
def term_2(Q2, e_f, g):
    return ((((1 - (m_Z**2 / Q2)) / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2)) *
            (1 - 4 * sin2th_w) / (4 * sin2th_w * (1- sin2th_w ))* e_f * g))
            
def term_3(Q2, e_f, g):
    return (1 / ((Q2 - m_Z**2)**2 + m_Z**2 * Gamma_Z**2) * 
            (1 + (1 - 4 * sin2th_w)**2) / (32 * sin2th_w**2 * (1-sin2th_w)**2)) * g**2

def summation_terms(Q2, e_f, g):
    return  (term_1(Q2, e_f) + term_2(Q2, e_f, g) + term_3(Q2, e_f, g))

def dsigma_y_sm(Q2, y, quark_couplings):
    tau = Q2 / s
    d_sigma = 0

    x1 = np.sqrt(tau) * np.exp(y)
    x2 = np.sqrt(tau) * np.exp(-y)
    if not(0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0):
        return 0.0

    for flavor, e_f, g_fR, g_fL in quark_couplings:
        sigmaprime = f_s(x1, tau, flavor, Q2)

        termL = summation_terms(Q2, e_f, g_fL)
        termR = summation_terms(Q2, e_f, g_fR)

        d_sigma += tau * (termL + termR) * sigmaprime

    d_sigmasm = factor * 0.389379 * 1e9 * d_sigma
    return d_sigmasm

def dsigma_y_liv(Q2, y, cL, cR, p1, p2, quark_couplings):
    tau = Q2 / s
    d_sigmaL = 0 
    d_sigmaR = 0

    x1 = np.sqrt(tau) * np.exp(y)
    x2 = np.sqrt(tau) * np.exp(-y)

    if not(0.0 <= x1 <= 1.0 and 0.0 <= x2 <= 1.0):
        return 0.0
    
    for flavor, e_f, g_fR, g_fL in quark_couplings:
        _, smeL_12 = sigma_hat_prime(x1, tau, cL, p1, p2, flavor, Q2)
        _, smeL_21 = sigma_hat_prime(x2, tau, cL, p2, p1, flavor, Q2)

        _, smeR_12 = sigma_hat_prime(x1, tau, cR, p1, p2, flavor, Q2)
        _, smeR_21 = sigma_hat_prime(x2, tau, cR, p2, p1, flavor, Q2)

        smeL = smeL_12 + smeL_21
        smeR = smeR_12 + smeR_21

        termL = summation_terms(Q2, e_f, g_fL)
        termR = summation_terms(Q2, e_f, g_fR)

        d_sigmaL += tau * smeL * termL
        d_sigmaR += tau * smeR * termR

    return factor *  0.389379 * 1e9 * (d_sigmaL + d_sigmaR)


    


