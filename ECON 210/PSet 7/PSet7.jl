# Solve Q1 of PSet 7. 

# set current folder as the working directory
cd(@__DIR__)

# ----------------------------------------------------------------------------- #

# Packages
using Plots
using QuantEcon

# ============================================================================= #

# Define Primitives:

## Utility function

function u(c, γ)
    if c <= 0.01
        return -100000
    # elseif γ == 1
    #     return log(c)
    else
        return c^(1 - γ) / (1 - γ)
    end
end

## Derivative of utility function

function u′(c, γ)
    if c <= 0.01
        return 1000
    # elseif γ == 1
    #     return 1/c
    else
        return c^(-γ)
    end
end

## Inverse of derivative of utility function

function u′_inv(MU, γ)
    return (MU)^(-1 / γ)
end

γ = 10 # risk aversion
β = 0.9 
μ_x = 4.7 
σ_x = sqrt(0.01)
R_f = 1.01 
μ_r = 0.08 
σ_r = sqrt(0.11)

# ============================================================================= #

grid_size = 100

# Set up the fixed grid for future assets.

b′_grid = exp.(range(0, log(1000), length = grid_size)) .- 1

# Set up the endogenous grid for consumption. 

c_grid = exp.(range(0, log(1000), length = grid_size)) .- 1

# Set up the grid for portfolio choice ϕ

ϕ_grid = range(0, 1, length = grid_size)

# ============================================================================= #

# Discretize the income process. 

num_income_states = 7
income_chain = tauchen(num_income_states, 0, σ_x, μ_x, N = 3)
income_states = exp.(income_chain.state_values)
income_probs = income_chain.P

# ============================================================================= #

# Discretize the returns process. 

num_return_states = 7
return_chain = tauchen(num_return_states, 0, σ_r, μ_r, N = 3)
return_states = exp.(return_chain.state_values)
return_probs = return_chain.P

# ============================================================================= #

# initial values 

w_0 = 0 
y_0 = 0 

# ============================================================================= #

# Define an initial value function.

V = zeros(grid_size)

# ============================================================================= #

# Define a function to calculate wealth distributions in the next period. 

function wealth_dist(ϕ, b′, c, income_states, income_probs, return_states, return_probs)

    
    
end
