using Distributions
using Plots
using QuantEcon

# =================================================================== #

# Define Primitives: 

## Utility function

function u(c, γ)
    if c <= 0.01 
        return -100000
    elseif γ == 1
        return log(c)
    else
        return c^(1 - γ) / (1 - γ)
    end
end

## Derivative of utility function

function u′(c, γ)
    if c <= 0.01 
        return 1000
    elseif γ == 1
        return 1/c
    else
        return c^(-γ)
    end
end

# =================================================================== #

# Define the grids we will use.

### Define the grid sizes

nx = 200 # number of grid points for cash-on-hand x
ns = 200 # number of grid points for saving s
ny = 200 # number of grid points for income y

### Define the grids.
### Note that the saving grid implicitly contains the borrowing constraint.

x_grid = range(10, 320, length = nx) # Grid for cash on hand
s_grid = range(0.0, 300, length = ns) # Grid for saving.

# =================================================================== #

# Define function to solve the infinite-horizon consumption savings problem with
# a borrowing constraint.

"""
    solve_csp(ρ, σ, μ, R, δ, x_grid, s_grid, ny, ψ)

Solve the infinite-horizon consumption savings problem with a borrowing constraint.
"""
function solve_csp(ρ, σ, μ, R, δ, x_grid, s_grid, ny, ψ) 

    # Define the grid sizes
    nx = length(x_grid)
    ns = length(s_grid)

    # Define the transition matrix for the AR(1) process
    Π = tauchen(ny, ψ, σ, μ, 3)

    # Create an initial guess for the value function
    V = zeros(nx, ns)

    # Create the policy functions. 
    s′ = zeros(nx, ns)
    c = zeros(nx, ns)

    # Define the objective function
    function objective(s, x, V, Π, R, δ, ρ)
        # Define the objective function
        function F(s′)
            # Compute the consumption
            c = R * s + x - s′

            # Compute the expected value function
            EV = 0.0
            for j in 1:ns
                EV += Π.p[j, s′] * V[x, j]
            end

            # Compute the objective function
            return - u(c, ρ) - δ * EV
        end
        return F
    end

    # Define the objective function
    F = objective(s, x, V, Π, R, δ, ρ)

    # Solve the problem
    for i in 1:nx
        for j in 1:ns
            # Solve the problem
            res = optimize(F, 0.0, x_grid[i])
            s′[i, j] = res.minimizer
            c[i, j] = R * s_grid[j] + x_grid[i] - s′[i, j]
        end
    end

    # Compute the value function
    for i in 1:nx
        for j in 1:ns
            # Compute the expected value function
            EV = 0.0
            for k in 1:ns
                EV += Π.p[k, s′[i, j]] * V[i, k]
            end

            # Compute the value function
            V[i, j] = u(c[i, j], ρ) + δ * EV

    # Compute the policy function

    

