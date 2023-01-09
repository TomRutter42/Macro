# set current folder as the working directory
cd(@__DIR__)

# source in Primitives.jl 
include("Primitives.jl")


## Specify the process for log(xt) - μ, then add μ back in and exponentiate.
## Use tauchen() from QuantEcon. 

### Specify the number of grid points for the discretization.

NTauchen = 15 # differs from the simulation script

## Assert than NTauchen_Sims is odd, throwing an error if not. 

odd_error_message = 
    "Number of grid points for discretization must be odd."

NTauchen % 2 == 1 ? nothing :
    throw(AssertionError(odd_error_message))

### Specify the number of standard deviations to include in the discretization.
### We want the lowest income state to be around 7000, as suggested by the Q. 
### Note sd of an AR(1) is σ_ε / sqrt(1 - ρ^2).
### log(7000) - log(50000) ≈ -2 
### and the standard deviation, given the parameters, is 
### σ_ε / sqrt(1 - ρ^2) = 0.12 / sqrt(1 - 0.95^2) ≈ 0.38
### 2 / 0.38 = 5.26

SDs = 5.1

### Specify the discretization.

log_x_minus_μ = tauchen(NTauchen, ρ, σ_ε, 0, SDs)

### Add μ back in, then exponentiate.

x_states = exp.(log_x_minus_μ.state_values .+ μ)
Π_x = log_x_minus_μ.p

# Stationary distribution - tested with higher powers. 
stat_dist = stationary_distributions(log_x_minus_μ)[1]