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

@njit
def foc_constrained(kprime, w_uncons,vprime_fn,phi,R,beta):
    """ 
    Fn to solve for unknown kprime
     Note: inputs scaler value for w_uncons
    """
    kstar = kstar_fn(R)
    output = uprime_fn(w_uncons - (1-phi)*kprime)*(1-phi) -beta*vprime_fn(f(kprime) -(R*phi*kprime))*(fprime(kprime)-R*phi)
    return output 

@njit
def foc_unconstrained(bprime, w_uncons,vprime_fn,phi,R,beta):
    """ 
    Fn to solve for unknown bprime
    Note: inputs scaler value for w_uncons
    """
    kstar = kstar_fn(R)
    output = uprime_fn(w_uncons- bprime - kstar) - beta*R*vprime_fn( f(kstar) +(R*bprime) )
    return output 

@njit
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

@njit
def solve_kprime_bprime(wgrid, phi, R, beta):

    # Form an initial guess for the value function 

    vprime_old = np.ones(len(wgrid))

    # Set up empty arrays to store the optimal kprime and bprime

    kprime = np.empty(len(wgrid))
    bprime = np.empty(len(wgrid))

    # Run the value function iteration algorithm. 

    ## First, we specify the tolerance level for the algorithm.

    tol = 1e-6

    ## Next, we set up a while loop that will run until the distance between the old and new value functions is less than the tolerance level.

    dist = 1

    while dist > tol:

        # Set up an empty array to store the new value function

        vprime_new = np.empty(len(wgrid))

        # Calulate the derivative of the old value function

        vprime_derivative = pchip(wgrid, vprime_old)

        # Use the threshold function to find the threshold w_star 

        w_star = opt.fsolve(threshold, 0.5, args=(vprime_derivative, phi, R, beta))

        # Solve for the optimal Vprime, kprime, bprime for w < w_star

        for i in range(len(wgrid)):

            if wgrid[i] < w_star:

                # Solve for the optimal kprime

                kprime[i] = opt.fsolve(foc_constrained, 0.5, args=(wgrid[i], vprime_derivative, phi, R, beta))

                # Solve for the optimal bprime

                bprime[i] = 0

            else:

                # Solve for the optimal bprime

                bprime[i] = opt.fsolve(foc_unconstrained, 0.5, args=(wgrid[i], vprime_derivative, phi, R, beta))

                # Solve for the optimal kprime

                kprime[i] = kstar_fn(R)

            # Calculate the new value function

            vprime_new[i] = utility_fn(wgrid[i] - bprime[i] - kprime[i]) + beta * vprime_old(wgrid[i] - bprime[i] - kprime[i])

        # Calculate the distance between the old and new value functions

        dist = np.max(np.abs(vprime_new - vprime_old))

        # Update the old value function with the new value function. 

        vprime_old = vprime_new

    return kprime, bprime, vprime_new











