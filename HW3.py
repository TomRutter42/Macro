# Packages 

import numpy as np
from numba import njit, int64, prange
import matplotlib.pyplot as plt

from numpy.polynomial import Polynomial
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator as pchip
import scipy.optimize as opt

# Set Parameters 

beta = 0.8 
R = 1.02
phi = 0.8

# Set Up Functions 

@njit
def utility_fn(c):
    """
    """
    u = np.log(c)
    return u 

@njit
def uprime_fn(c):
    """
    """
    u = 1/c
    return u 

@njit
def f(k):
    """
    """
    y = k**(1/3)
    return y 

@njit
def fprime(k):
    """
    """
    return (1/3)*k**(-2/3)

@njit
def kstar_fn(R):
    """
    """
    return (R*3)**(-3/2) 

# -----------------------------------------------------------------------------

# Define FOCs 

def foc_constrained(kprime, w_uncons,vprime_fn,phi,R,beta):
    """ 
    Fn to solve for unknown kprime
     Note: inputs scaler value for w_uncons
    """
    kstar = kstar_fn(R)
    output = uprime_fn(w_uncons - (1-phi)*kprime)*(1-phi) -beta*vprime_fn(f(kprime) -(R*phi*kprime))*(fprime(kprime)-R*phi)
    return output 

def foc_unconstrained(bprime, w_uncons,vprime_fn,phi,R,beta):
    """ 
    Fn to solve for unknown bprime
    Note: inputs scaler value for w_uncons
    """
    kstar = kstar_fn(R)
    output = uprime_fn(w_uncons- bprime - kstar) - beta*R*vprime_fn( f(kstar) +(R*bprime) )
    return output 

def threshold(wbar,vprime_fn,phi,R,beta):
    """
    Function to solve for unknown threshold wbar
    """
    kstar = kstar_fn(R)
    output = uprime_fn(wbar-(1-phi)*kstar) - beta*R*vprime_fn(kstar**(1/3) - R*phi*kstar)
    return output 

# -----------------------------------------------------------------------------

# Set up discrete grid for w

w_grid = np.linspace(0.5, 5, 10)

# -----------------------------------------------------------------------------

# Define a function that solves for the optimal kprime and bprime given w. 

def solve_kprime_bprime(wgrid_arg, phi_arg, R_arg, beta_arg):

    # Form an initial guess for the value function 

    vprime = np.ones(len(wgrid_arg))

    # Calculate 









