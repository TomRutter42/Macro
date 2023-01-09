# This file sets up stuff that is sourced in by subsequent files. 

# set current folder as the working directory
cd(@__DIR__)

# --------------------- #

# Packages
using Interpolations
using JLD
using Plots
using QuantEcon
using Random # for seed for reproducible income simulations in Q1
using StatsBase # for sampling with given probabilities, used in simulations in Q1

# ---------------------

# Utility functions

function u(c, γ)
    if γ == 1
        return log(c)
    else
        return c^(1 - γ) / (1 - γ)
    end
end

### Sanity check: 

utility_error_message = 
    "Something wrong with instantaneous utility function."
u(1, 2) == -1.0 ? nothing : 
    throw(AssertionError(utility_error_message))

## Derivative of utility function

function u′(c, γ)
    if γ == 1
        return 1/c
    else
        return c^(-γ)
    end
end

## Inverse of derivative of utility function

function u′_inv(MU, γ)
    return (MU)^(-1 / γ)
end

# ---------------------

## Specify Parameters 

β = 0.95 
# β = 0.1 # test - liquidity binds more clearly. 
γ = 4.0 
R = 1.01
R_c = 1.0379 # for question 3
Lbar = 40000.0 # for question 3
L_t = Lbar * (R_c - 1) * R_c^10 / (R_c^10 - 1) # for question 3
T = 55
life_begins = 25

# ---------------------------------------------------- #

## Specify deterministic component of income by age. 

A = zeros(T)

for t in 0:35 

    # t + 1 since Julia is 1-indexed, 
    # not 0-indexed like Python. 
    A[t + 1] = (-3.12 + 0.26 * (t + 25) - 0.0024 * (t + 25)^2) / 1.88
    
end

for t in 36:(T - 1)

    # t + 1 since Julia is 1-indexed, 
    # not 0-indexed like Python. 
    A[t + 1] = ( -3.12 + 0.26 * 60 - 0.0024 * 60^2 ) / 1.88  

end

## Specify deterministic component of income by age. 

A = zeros(T)

for t in 0:35 

    # t + 1 since Julia is 1-indexed, 
    # not 0-indexed like Python. 
    A[t + 1] = (-3.12 + 0.26 * (t + 25) - 0.0024 * (t + 25)^2) / 1.88
    
end

for t in 36:(T - 1)

    # t + 1 since Julia is 1-indexed, 
    # not 0-indexed like Python. 
    A[t + 1] = ( -3.12 + 0.26 * 60 - 0.0024 * 60^2 ) / 1.88  

end

# ---------------------------------------------------- #

# Specfify the discretization of the stochastic income process. 

## yt = xt At 
## log(xt) - μ = ρ * (log(x(t-1)) - μ) + εt

## Specify parameters of the AR(1) process.

μ = log(50000.0)
ρ = 0.95 
# ρ = 0.0 # test - lines should line up. 
σ_ε = 0.12

# ---------------------------------------------------- #

# Grid Settings 

## number of gridpoints. 
n = 2000

a_max = 1000000.0 ## 1 million

a′_grid = exp.(range(0.0, log(a_max), length = n)) .- 1

# ---------------------------------------------------- #

# Initialize policy function and value function arrays. 
C = zeros(n, NTauchen, T)
V = zeros(n, NTauchen, T)